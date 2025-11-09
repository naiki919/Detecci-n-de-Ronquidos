#!/usr/bin/env python3
import argparse, os, wave, datetime as dt, time
import numpy as np
from collections import deque
from scipy.signal import stft, get_window
from scipy.fftpack import dct

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

import serial

HDR0, HDR1 = 0xAA, 0x55

def find_sync(port):
    while True:
        b = port.read(1)
        if not b: continue
        if b[0] == HDR0:
            b2 = port.read(1)
            if b2 and b2[0] == HDR1:
                return True

def read_packet(port):
    if not find_sync(port): return None
    ln = port.read(2)
    if len(ln) < 2: return None
    n = ln[0] | (ln[1] << 8)
    payload = port.read(n * 2)
    if len(payload) < n * 2: return None
    return np.frombuffer(payload, dtype='<i2')

def load_tflite(path):
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    return it, it.get_input_details()[0], it.get_output_details()[0]

def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
def mel_to_hz(m): return 700.0 * (10.0**(m / 2595.0) - 1.0)

def mel_filterbank(sr, n_fft, n_mels, fmin=80.0, fmax=6000.0):
    n_freqs = n_fft // 2 + 1
    m_min, m_max = hz_to_mel(fmin), hz_to_mel(min(fmax, sr/2.0))
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts)
    bins = np.floor((n_fft + 1) * f_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f0, fm, f1 = bins[m - 1], bins[m], bins[m + 1]
        f0 = np.clip(f0, 0, n_freqs - 1)
        fm = np.clip(fm, 0, n_freqs - 1)
        f1 = np.clip(f1, 0, n_freqs - 1)
        if fm == f0: fm = min(fm + 1, n_freqs - 1)
        if f1 == fm: f1 = min(f1 + 1, n_freqs - 1)
        if f0 < fm:
            fb[m-1, f0:fm] = (np.arange(f0, fm) - f0) / (fm - f0)
        if fm < f1:
            fb[m-1, fm:f1] = (f1 - np.arange(fm, f1)) / (f1 - fm)
    return fb

def compute_mfcc_window(x_f32, sr, n_fft=512, hop=160, n_mels=40, n_mfcc=20,
                        fmin=80.0, fmax=6000.0, target_frames=150):
    win = get_window("hann", n_fft, fftbins=True)
    _, _, Z = stft(x_f32, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft - hop,
                   nfft=n_fft, boundary="zeros", padded=True)
    S_pow = (np.abs(Z) ** 2).astype(np.float32)
    fb = mel_filterbank(sr, n_fft, n_mels, fmin=fmin, fmax=fmax)
    S_mel = np.dot(fb, S_pow)                  # [n_mels, time]
    S_log = np.log(np.maximum(S_mel, 1e-10))   # log natural
    M = dct(S_log, type=2, axis=0, norm='ortho')
    MFCC = M[:n_mfcc, :].T                     # [frames, 20]

    T = MFCC.shape[0]
    if T < target_frames:
        pad = np.zeros((target_frames - T, n_mfcc), dtype=np.float32)
        MFCC = np.vstack([MFCC, pad])
    elif T > target_frames:
        MFCC = MFCC[:target_frames, :]

    mean = MFCC.mean(axis=0, keepdims=True)
    std = MFCC.std(axis=0, keepdims=True)
    MFCC = (MFCC - mean) / (std + 1e-8)
    return MFCC.astype(np.float32)

def get_hw(det):
    s = det['shape']
    if len(s) == 4:
        return (int(s[1]), int(s[2])) if s[1] > 1 and s[2] > 1 else (int(s[2]), int(s[3]))
    if len(s) == 2:
        return (int(s[1]), 1)
    return (150, 20)

def q_in(x, det):
    dt = det['dtype']
    if dt == np.float32: return x.astype(np.float32)
    scale, zp = det.get('quantization', (1.0, 0))
    if dt == np.int8:  return (x / scale + zp).astype(np.int8)
    if dt == np.uint8: return (x / scale + zp).astype(np.uint8)
    raise ValueError("dtype de entrada no soportado")

def dq_out(y, det):
    dt = det['dtype']
    if dt == np.float32: return y.astype(np.float32)
    scale, zp = det.get('quantization', (1.0, 0))
    return scale * (y.astype(np.float32) - zp)

def softmax(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    e = np.exp(v - np.max(v))
    s = e.sum()
    return e / s if s > 0 else np.zeros_like(v)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--serial', default='/dev/ttyACM0')
    ap.add_argument('--sr', type=int, default=16000)
    ap.add_argument('--win', type=float, default=1.5)
    ap.add_argument('--hop', type=float, default=0.5)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--outdir', default='data')
    ap.add_argument('--pos-index', type=int, default=1)
    ap.add_argument('--avg-k', type=int, default=5, help='frames para promedio móvil')
    ap.add_argument('--hyst', type=float, default=0.1, help='histéresis: bajar de (thr - hyst) para reset')
    ap.add_argument('--cooldown', type=float, default=2.0, help='segundos entre beeps')
    ap.add_argument('--beep-ms', type=int, default=600, help='duración del beep solicitado al Arduino')
    ap.add_argument('--log-raw', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    raw_dir = os.path.join(args.outdir, "raw"); os.makedirs(raw_dir, exist_ok=True)
    events_csv = os.path.join(args.outdir, "events.csv")
    if not os.path.exists(events_csv):
        with open(events_csv, 'w', encoding='utf-8') as f:
            f.write("timestamp,score,threshold\n")

    it, in_det, out_det = load_tflite(args.model)
    inp_idx = it.get_input_details()[0]['index']
    out_idx = it.get_output_details()[0]['index']
    H, W = get_hw(in_det)
    print(f"[INFO] Modelo: input {H}x{W}")

    ser = serial.Serial(args.serial, 115200, timeout=1)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = os.path.join(raw_dir, f"mic_{ts}.wav")
    wf = wave.open(wav_path, 'wb'); wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(args.sr)

    win_n = int(args.win * args.sr)
    hop_n = int(args.hop * args.sr)
    circ = deque(maxlen=win_n); [circ.append(0) for _ in range(win_n)]
    acc = 0
    printed_raw = False

    # estados de detección
    state_on = False
    last_event_time = 0.0
    scores = deque(maxlen=max(1, args.avg_k))

    try:
        print("[INFO] Ejecutando... Ctrl+C para salir")
        while True:
            s = read_packet(ser)
            if s is None: continue
            wf.writeframes(s.tobytes())
            for v in s: circ.append(int(v))
            acc += len(s)

            while acc >= hop_n:
                acc -= hop_n
                x = np.array(circ, dtype=np.int16).astype(np.float32)
                mx = np.max(np.abs(x))
                if mx > 0: x = x / mx

                MFCC = compute_mfcc_window(
                    x, args.sr, n_fft=512, hop=160, n_mels=40, n_mfcc=20,
                    fmin=80.0, fmax=6000.0, target_frames=150
                )

                tin = MFCC[None, :, :, None]
                it.set_tensor(inp_idx, q_in(tin, in_det))
                it.invoke()
                y = dq_out(it.get_tensor(out_idx), out_det)
                v = np.asarray(y).squeeze().reshape(-1)
                if (v.min() < 0.0) or (v.max() > 1.0) or not (0.98 <= float(v.sum()) <= 1.02):
                    v = softmax(v)
                score = float(v[min(max(args.pos_index,0), len(v)-1)])

                scores.append(score)
                avg = float(np.mean(scores))
                print(f"[INF] score={score:.3f} avg={avg:.3f}")

                now = time.time()
                if not state_on:
                    # activación: supera umbral con promedio
                    if avg >= args.threshold and (now - last_event_time) >= args.cooldown:
                        state_on = True
                        last_event_time = now
                        # log evento
                        with open(events_csv, 'a', encoding='utf-8') as f:
                            f.write(f"{dt.datetime.now().isoformat(timespec='seconds')},{avg:.4f},{args.threshold}\n")
                        # enviar beep con duración (byte 'B' seguido de 2 bytes LE de ms)
                        try:
                            ms = max(1, min(5000, int(args.beep_ms)))
                            ser.write(b'B')
                            ser.write(bytes([ms & 0xFF, (ms >> 8) & 0xFF]))
                        except Exception as e:
                            print("[WARN] buzzer:", e)
                else:
                    # desactivación: baja por debajo de (thr - hyst)
                    if avg < (args.threshold - args.hyst):
                        state_on = False

    except KeyboardInterrupt:
        print("\n[SALIR]")
    finally:
        try: wf.close()
        except: pass
        try: ser.close()
        except: pass
        print(f"[INFO] Audio: {wav_path}")
        print(f"[INFO] Eventos: {events_csv}")

if __name__ == "__main__":
    main()
