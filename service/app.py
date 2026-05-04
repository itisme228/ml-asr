import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(current_dir)

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from flask import Flask, render_template, request, jsonify
app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, 'templates'),
    static_folder=os.path.join(ROOT_DIR, 'static')
)

STATIC_AUDIO_DIR = os.path.join(ROOT_DIR, "static", "audio")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(STATIC_AUDIO_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

import math
import uuid
import atexit
import shutil
import gc
import torch
import torchaudio.transforms as T
import soundfile as sf
import time
from flask import Flask, render_template, request, jsonify
from torchaudio.utils import _download_asset

from models.BSRNN import BSRNN
from models.FRCRN import FRCRN
from models.DCCRN import DCCRN
from models.deepfilternet import DeepFilterNet2
from models.fullsubnet import FullSubNet
from models.PHASEN import PHASEN
from models.SGMSE import SGMSE
from models.metricgan import metricgan
from models.dcunet import DCUNET

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# --- Конфигурация ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
N_FFT = 512
HOP_LENGTH = 256

SGMSE_ALPHA = 0.5
SGMSE_T_EPS = 0.03
SGMSE_SIGMA_MIN = 0.05
SGMSE_SIGMA_MAX = 0.5
SGMSE_GAMMA = 1.5
SGMSE_N_STEPS = 30

current_model_name = None
current_model = None

def cleanup_static_files():
    folder = STATIC_AUDIO_DIR
    print(f"[*] Завершение работы: очистка папки {folder}...")
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path) # Удаляем файл или ссылку
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path) # Удаляем подпапку, если она есть
            except Exception as e:
                print(f'[-] Не удалось удалить {file_path}. Причина: {e}')

def compress_stft(stft, alpha=SGMSE_ALPHA):
    mag = torch.abs(stft)
    phase = torch.angle(stft)
    return (mag ** alpha) * torch.exp(1j * phase)

def decompress_stft(stft, alpha=SGMSE_ALPHA):
    mag = torch.abs(stft)
    phase = torch.angle(stft)
    return (mag ** (1.0 / alpha)) * torch.exp(1j * phase)

def sde_g(t):
    return SGMSE_SIGMA_MIN * (SGMSE_SIGMA_MAX / SGMSE_SIGMA_MIN) ** t * math.sqrt(2 * math.log(SGMSE_SIGMA_MAX / SGMSE_SIGMA_MIN))

def get_model(model_name):
    global current_model_name, current_model

    if current_model_name == model_name and current_model is not None:
        return current_model

    print(f"[*] Смена модели. Выгрузка {current_model_name} и загрузка {model_name}...")

    if current_model is not None:
        del current_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if model_name == "BSRNN":
        model = BSRNN(freq_bins=257, feature_dim=128, hidden_size=256, num_layers=2)
    elif model_name == "FRCRN":
        model = FRCRN(n_fft=640, hop_len=160, win_len=320)
    elif model_name == "DCCRN":
        model = DCCRN()
    elif model_name == "DeepFilterNet2":
        model = DeepFilterNet2(n_fft=N_FFT, sr=SR, n_erb=32)
    elif model_name == "FullSubNet":
        model = FullSubNet(num_freqs=257, N=15)
    elif model_name == "metricgan":
        model = metricgan(input_dim=257, hidden_dim=200, num_layers=2)
    elif model_name == "PHASEN":
        model = PHASEN()
    elif model_name == "DCUNET":
        model = DCUNET(n_fft=1024, hop_length=256)
    elif model_name == "ScoreModelDCUNet":
        model = SGMSE()
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    model_path = os.path.join(MODELS_DIR, f"{model_name.lower()}_weights.pth")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=False)
        print(f"[+] Веса успешно загружены из {model_path}")
    else:
        print(f"[-] ВНИМАНИЕ: Файл {model_path} не найден.")

    model.to(DEVICE)
    model.eval()

    current_model = model
    current_model_name = model_name
    return model

print("Загрузка ассетов шума...")
babble_path = _download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
rir_path = _download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")

data, samplerate = sf.read(babble_path)
BABBLE_WAVEFORM = torch.from_numpy(data).float().t()
if BABBLE_WAVEFORM.ndim == 1:
    BABBLE_WAVEFORM = BABBLE_WAVEFORM.unsqueeze(0)
BABBLE_WAVEFORM = T.Resample(samplerate, SR)(BABBLE_WAVEFORM.mean(dim=0, keepdim=True)).to(DEVICE)

rir_data, rir_samplerate = sf.read(rir_path)
RIR_WAVEFORM = torch.from_numpy(rir_data).float().t()
if RIR_WAVEFORM.ndim == 1:
    RIR_WAVEFORM = RIR_WAVEFORM.unsqueeze(0)
RIR_WAVEFORM = T.Resample(rir_samplerate, SR)(RIR_WAVEFORM.mean(dim=0, keepdim=True))
RIR_WAVEFORM = RIR_WAVEFORM[:, :int(SR * 0.3)].to(DEVICE)
RIR_WAVEFORM = RIR_WAVEFORM / torch.norm(RIR_WAVEFORM, p=2)

def get_snr_scale(signal, noise, snr_db=5):
    sig_power = signal.norm(p=2)**2 / (signal.numel() + 1e-8)
    noise_power = noise.norm(p=2)**2 / (noise.numel() + 1e-8)
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    return torch.sqrt(target_noise_power / (noise_power + 1e-8))

def apply_noise_to_tensor(clean, noise_type):
    n_len = clean.shape[-1]

    if noise_type == 'babble':
        noise = BABBLE_WAVEFORM
        if noise.shape[-1] < n_len:
            repeats = (n_len // noise.shape[-1]) + 2
            noise = noise.repeat(1, repeats)
        noise_crop = noise[:, :n_len]
        scale = get_snr_scale(clean, noise_crop, snr_db=5)
        noisy = clean + noise_crop * scale

    elif noise_type == 'rir':
        rir = RIR_WAVEFORM
        n_fft_conv = n_len + rir.shape[-1] - 1
        clean_fft = torch.fft.rfft(clean, n=n_fft_conv)
        rir_fft = torch.fft.rfft(rir, n=n_fft_conv)
        augmented = torch.fft.irfft(clean_fft * rir_fft, n=n_fft_conv)
        noisy = augmented[:, :n_len]
        white = torch.randn_like(clean)
        scale = get_snr_scale(noisy, white, snr_db=15)
        noisy = noisy + white * scale

    elif noise_type == 'white':
        noise = torch.randn(1, n_len, device=DEVICE)
        noise = noise / (noise.abs().max() + 1e-8)
        scale = get_snr_scale(clean, noise, snr_db=5)
        noisy = clean + noise * scale
    else:
        noisy = clean

    max_val = noisy.abs().max()
    if max_val > 1.0:
        noisy = noisy / (max_val + 1e-8)

    return noisy

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'Аудио файл не найден'}), 400

    noise_type = request.form.get('noise_type', 'white')
    model_type = request.form.get('model_type', 'BSRNN') # Добавлен параметр модели
    audio_file = request.files['audio']

    session_id = str(uuid.uuid4())[:8]
    raw_filename = f"{session_id}_raw.wav"
    raw_full_path = os.path.join(STATIC_AUDIO_DIR, raw_filename)

    start_time = time.time()
    try:
        audio_file.save(raw_full_path)
        data, samplerate = sf.read(raw_full_path)
        waveform = torch.from_numpy(data).float().t()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if samplerate != SR:
            waveform = T.Resample(samplerate, SR)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        clean_tensor = waveform.to(DEVICE)
        noisy_tensor = apply_noise_to_tensor(clean_tensor, noise_type)
        n_len = noisy_tensor.shape[-1]

        model = get_model(model_type)

        with torch.no_grad():
            if model_type == "BSRNN":
                X_complex = torch.stft(noisy_tensor, n_fft=N_FFT, hop_length=HOP_LENGTH, return_complex=False)
                S_hat = model(X_complex)
                S_hat_complex = torch.view_as_complex(S_hat)
                denoised_tensor = torch.istft(S_hat_complex.squeeze(0), n_fft=N_FFT, hop_length=HOP_LENGTH, length=n_len)

            elif model_type == "FRCRN":
                x_in = noisy_tensor.unsqueeze(0) 
                denoised_tensor = model(x_in).squeeze()

            elif model_type == "DCCRN":
                denoised_tensor = model(noisy_tensor).squeeze()

            elif model_type == "DeepFilterNet2":
                X_complex = torch.stft(
                    noisy_tensor, 
                    n_fft=N_FFT, 
                    hop_length=HOP_LENGTH, 
                    return_complex=True
                )
                
                S_hat_complex = model(X_complex)
                
                denoised_tensor = torch.istft(
                    S_hat_complex.squeeze(0), 
                    n_fft=N_FFT, 
                    hop_length=HOP_LENGTH, 
                    length=n_len
                )

            elif model_type == "FullSubNet":
                X_complex = torch.stft(noisy_tensor, n_fft=N_FFT, hop_length=HOP_LENGTH, return_complex=True)
                mag = torch.abs(X_complex)
                cirm = model(mag) 
                cirm = cirm.permute(0, 2, 1, 3) 
                real = X_complex.real * cirm[..., 0] - X_complex.imag * cirm[..., 1]
                imag = X_complex.real * cirm[..., 1] + X_complex.imag * cirm[..., 0]
                S_hat_complex = torch.complex(real, imag)
                denoised_tensor = torch.istft(S_hat_complex.squeeze(0), n_fft=N_FFT, hop_length=HOP_LENGTH, length=n_len)       

            elif model_type == "metricgan":
                X_complex = torch.stft(noisy_tensor, n_fft=N_FFT, hop_length=HOP_LENGTH, return_complex=True)
                mag = torch.abs(X_complex).transpose(1, 2)
                mag_hat = model(mag).transpose(1, 2)
                phase = torch.angle(X_complex)
                S_hat_complex = mag_hat * torch.exp(1j * phase)
                denoised_tensor = torch.istft(S_hat_complex.squeeze(0), n_fft=N_FFT, hop_length=HOP_LENGTH, length=n_len)

            elif model_type == "PHASEN":
                X_complex = torch.stft(noisy_tensor, n_fft=N_FFT, hop_length=HOP_LENGTH, return_complex=True)
                x_in = torch.stack([X_complex.real, X_complex.imag], dim=1).transpose(2, 3) 
                out = model(x_in) 
                out_real = out[:, 0].transpose(1, 2)
                out_imag = out[:, 1].transpose(1, 2)
                S_hat_complex = torch.complex(out_real, out_imag)
                denoised_tensor = torch.istft(S_hat_complex.squeeze(0), n_fft=N_FFT, hop_length=HOP_LENGTH, length=n_len)

            elif model_type == "DCUNET":
                denoised_tensor = model(noisy_tensor).squeeze()

            elif model_type == "SpeechEnhancementPipeline":
                denoised_tensor = model(noisy_tensor.unsqueeze(0)).squeeze()

            elif model_type == "ScoreModelDCUNet":
                window = torch.hann_window(N_FFT).to(DEVICE)
                noisy_wav_1d = noisy_tensor.squeeze()
                Y = torch.stft(noisy_wav_1d, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True, center=True)
                Y_c = compress_stft(Y, alpha=SGMSE_ALPHA).unsqueeze(0).unsqueeze(0)
                z = (torch.randn_like(Y_c) + 1j * torch.randn_like(Y_c)) / math.sqrt(2)
                x = Y_c + SGMSE_SIGMA_MAX * z
                t_steps = torch.linspace(1.0, SGMSE_T_EPS, SGMSE_N_STEPS, device=DEVICE)
                dt = (1.0 - SGMSE_T_EPS) / SGMSE_N_STEPS

                for i in range(SGMSE_N_STEPS):
                    t_val = t_steps[i].expand(1)
                    score = model(x, t_val, Y_c)
                    gamma_val = SGMSE_GAMMA
                    g_val = sde_g(t_val).view(-1, 1, 1, 1)
                    drift = gamma_val * (Y_c - x) - (g_val**2) * score
                    z_step = (torch.randn_like(x) + 1j * torch.randn_like(x)) / math.sqrt(2)
                    diffusion = g_val * math.sqrt(dt) * z_step if i < SGMSE_N_STEPS - 1 else 0
                    x = x - drift * dt + diffusion

                S_hat_c = x.squeeze()
                S_hat = decompress_stft(S_hat_c, alpha=SGMSE_ALPHA)
                denoised_tensor = torch.istft(S_hat, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, center=True)
                
                if denoised_tensor.shape[-1] > n_len:
                    denoised_tensor = denoised_tensor[..., :n_len]
                elif denoised_tensor.shape[-1] < n_len:
                    import torch.nn.functional as F
                    denoised_tensor = F.pad(denoised_tensor, (0, n_len - denoised_tensor.shape[-1]))
                    
                denoised_tensor = denoised_tensor.unsqueeze(0)

        def normalize(tensor):
            abs_max = tensor.abs().max()
            if abs_max > 1.0:
                return tensor / (abs_max + 1e-8)
            return tensor

        denoised_tensor = normalize(denoised_tensor)

        paths = {
            'clean': f"static/audio/{session_id}_clean.wav",
            'noisy': f"static/audio/{session_id}_noisy_{noise_type}.wav",
            'denoised': f"static/audio/{session_id}_{model_type}_denoised.wav"
        }

        sf.write(os.path.join(ROOT_DIR, paths['clean']), clean_tensor.squeeze().cpu().numpy(), SR)
        sf.write(os.path.join(ROOT_DIR, paths['noisy']), noisy_tensor.squeeze().cpu().numpy(), SR)
        sf.write(os.path.join(ROOT_DIR, paths['denoised']), denoised_tensor.squeeze().cpu().numpy(), SR)

        print(f"Обработка моделью {model_type} завершена за {time.time() - start_time:.2f} сек.")
        return jsonify(paths)

    except Exception as e:
        print(f"ОШИБКА: {e}")
        return jsonify({'error': str(e)}), 500

atexit.register(cleanup_static_files)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    