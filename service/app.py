import os
import shutil
import uuid
import torch
import torchaudio.transforms as T
import soundfile as sf
import time
from flask import Flask, render_template, request, jsonify
from torchaudio.utils import _download_asset
from models.SGMSE import BSRNN

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__, static_folder='static')
os.makedirs("static/audio", exist_ok=True)

# --- Конфигурация ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
N_FFT = 512
HOP_LENGTH = 256
MODEL_PATH = "models/bsrnn_weights_final.pth"

# --- Загрузка модели ---
print("Загрузка модели...")
model = BSRNN(freq_bins=257, feature_dim=128, hidden_size=256, num_layers=2).to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Модель загружена.")
else:
    print(f"ВНИМАНИЕ: Файл {MODEL_PATH} не найден. Прогнозы будут случайными.")

# --- Загрузка шумов ---
print("Загрузка ассетов шума...")
babble_path = _download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
rir_path = _download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")

data, samplerate = sf.read(babble_path)
BABBLE_WAVEFORM = torch.from_numpy(data).float().t()
if BABBLE_WAVEFORM.ndim == 1:
    BABBLE_WAVEFORM = BABBLE_WAVEFORM.unsqueeze(0)
sr_b = samplerate
BABBLE_WAVEFORM = T.Resample(sr_b, SR)(BABBLE_WAVEFORM.mean(dim=0, keepdim=True)).to(DEVICE)

rir_data, rir_samplerate = sf.read(rir_path)
RIR_WAVEFORM = torch.from_numpy(rir_data).float().t()
if RIR_WAVEFORM.ndim == 1:
    RIR_WAVEFORM = RIR_WAVEFORM.unsqueeze(0)
sr_r = rir_samplerate
RIR_WAVEFORM = T.Resample(sr_r, SR)(RIR_WAVEFORM.mean(dim=0, keepdim=True))
RIR_WAVEFORM = RIR_WAVEFORM[:, :int(SR * 0.3)].to(DEVICE)
RIR_WAVEFORM = RIR_WAVEFORM / torch.norm(RIR_WAVEFORM, p=2)

def clear_audio_folder():
    """Удаляет все файлы из папки static/audio"""
    folder = 'static/audio'
    print(f"\n[Cleanup] Очистка временных файлов в {folder}...")
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Не удалось удалить {file_path}. Причина: {e}')
    print("[Cleanup] Готово. Приложение закрыто.")

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
        # Добавляем немного белого шума для реалистичности
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
    audio_file = request.files['audio']
    
    session_id = str(uuid.uuid4())[:8]
    # ВНИМАНИЕ: Если браузер шлет WebM, soundfile его не прочитает. 
    # Лучше на фронтенде слать WAV или использовать библиотеку для конвертации.
    raw_path = f"static/audio/{session_id}_raw.wav" 
    audio_file.save(raw_path)

    start_time = time.time()
    print("Начинаю обработку нейросетью...")

    try:
        # Читаем файл
        data, samplerate = sf.read(raw_path)
        waveform = torch.from_numpy(data).float().t()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
            
        sr = samplerate
        if sr != SR:
            waveform = T.Resample(sr, SR)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        clean_tensor = waveform.to(DEVICE)
        
        # 1. Применяем шум
        noisy_tensor = apply_noise_to_tensor(clean_tensor, noise_type)
        
        # 2. Очистка (Denoising)
        with torch.no_grad():
            noisy_batched = noisy_tensor.unsqueeze(0) 
            X_complex = torch.stft(noisy_batched.squeeze(1), n_fft=N_FFT, hop_length=HOP_LENGTH, return_complex=False)
            
            S_hat = model(X_complex)
            
            S_hat_complex = torch.view_as_complex(S_hat)
            denoised_batched = torch.istft(S_hat_complex, n_fft=N_FFT, hop_length=HOP_LENGTH, length=noisy_tensor.shape[-1])
            denoised_tensor = denoised_batched.squeeze(0)

        # --- НОРМАЛИЗАЦИЯ (Важно для проигрывания) ---
        def normalize(tensor):
            abs_max = tensor.abs().max()
            if abs_max > 1.0:
                return tensor / (abs_max + 1e-8)
            return tensor

        denoised_tensor = normalize(denoised_tensor)

        paths = {
        'clean': f"/static/audio/{session_id}_clean.wav",
        'noisy': f"/static/audio/{session_id}_noisy_{noise_type}.wav",
        'denoised': f"/static/audio/{session_id}_denoised.wav"
        }   

        # Сохраняем (soundfile ожидает [samples, channels] для numpy)
        sf.write(paths['clean'].lstrip('/'), clean_tensor.squeeze().cpu().numpy(), SR)
        sf.write(paths['noisy'].lstrip('/'), noisy_tensor.squeeze().cpu().numpy(), SR)
        sf.write(paths['denoised'].lstrip('/'), denoised_tensor.squeeze().cpu().numpy(), SR)

        end_time = time.time()
        print(f"Обработка завершена за {end_time - start_time:.2f} сек.")

        return jsonify(paths) # Теперь return в конце

    except Exception as e:
        print(f"ОШИБКА: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import atexit
    atexit.register(clear_audio_folder)
    app.run(debug=True, port=5000)