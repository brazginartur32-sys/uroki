"""Звук с нуля: грустная «slowed + reverb» музыка в соль-диез миноре (как тональность референса),
два «вокальных» всплеска по громкости (как в референсе), тихие звуки Telegram,
цифровой глитч в конце и тишина после обрыва.

python3 audio.py timeline.json audio.wav
"""
import json
import sys
import wave

import numpy as np

SR = 44100
rng = np.random.default_rng(5)
tl = json.load(open(sys.argv[1]))
T = tl['t']
OUT = sys.argv[2]
DUR = tl['duration']
N = int(DUR * SR)

BPM = 76
BEAT = 60 / BPM
BAR = 4 * BEAT


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def place(buf, sig, t, gain=1.0):
    i = int(t * SR)
    if i >= len(buf) or i + len(sig) <= 0:
        return
    if i < 0:
        sig, i = sig[-i:], 0
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i] * gain


def fft_filter(x, lo=None, hi=None, order=2):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    H = np.ones_like(f)
    if hi:
        H *= 1 / np.sqrt(1 + (f / hi) ** (2 * order))
    if lo:
        H *= 1 / np.sqrt(1 + (lo / np.maximum(f, 1e-3)) ** (2 * order))
    return np.fft.irfft(X * H, len(x))


def piano(note, dur, vel=0.7, bright=0.8):
    f0 = midi(note)
    L = int((dur + 2.0) * SR)
    t = np.arange(L) / SR
    out = np.zeros(L)
    for k in range(1, 10):
        fk = k * f0 * np.sqrt(1 + 0.0004 * k * k)
        if fk > 8000:
            break
        amp = (1 / k ** 1.3) * (bright if k > 2 else 1)
        dec = 0.7 + 0.5 * k + f0 / 1000
        for det in (-0.7, 0.7):
            out += amp * 0.5 * np.sin(2 * np.pi * fk * (1 + det / 1730) * t + rng.random() * 6.28) * np.exp(-t * dec)
    att = np.minimum(1, t / 0.005)
    rel = np.clip(1 - (t - dur) / 0.5, 0, 1)
    return out * att * rel * vel * 0.3


def pad(notes, dur, vel=0.5):
    L = int((dur + 1.0) * SR)
    t = np.arange(L) / SR
    out = np.zeros(L)
    for n in notes:
        f = midi(n)
        for det in (-6, 0, 6):
            ff = f * 2 ** (det / 1200)
            # мягкая «пила» из нескольких гармоник
            for k in range(1, 6):
                out += np.sin(2 * np.pi * ff * k * t + rng.random() * 6.28) / (k * 1.6)
    env = np.minimum(1, t / 0.8) * np.clip(1 - (t - dur) / 0.9, 0, 1)
    return out * env * vel * 0.02


def lead(notes, vel=0.6):
    """«голос»: синус с вибрато и мягкой атакой; notes = [(midi, start, dur)]"""
    end = max(s + d for _, s, d in notes) + 1.0
    L = int(end * SR)
    t = np.arange(L) / SR
    freq = np.zeros(L)
    amp = np.zeros(L)
    for n, s, d in notes:
        i, j = int(s * SR), int((s + d) * SR)
        freq[i:j] = midi(n)
        tt = t[i:j] - s
        amp[i:j] = np.minimum(1, tt / 0.12) * np.clip(1 - (tt - d + 0.25) / 0.25, 0, 1)
    # плавные переходы высоты (портаменто)
    k = int(0.06 * SR)
    fs = np.convolve(np.where(freq > 0, freq, np.nan_to_num(freq)), np.ones(k) / k, mode='same')
    fs[fs <= 0] = 1
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.2 * t) * np.minimum(1, t / 0.6)
    ph = 2 * np.pi * np.cumsum(fs * vib) / SR
    s = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)
    breath = fft_filter(rng.normal(0, 1, L), lo=1500, hi=5000) * 0.05
    return (s + breath) * amp * vel * 0.25


def reverb(x, length=3.5, decay=1.1, mix=0.45, seed=0):
    r = np.random.default_rng(seed)
    L = int(length * SR)
    t = np.arange(L) / SR
    ir = r.normal(0, 1, L) * np.exp(-t / decay)
    ir = fft_filter(ir, hi=6000)
    ir[: int(0.02 * SR)] = 0
    ir /= np.sqrt((ir ** 2).sum())
    n = 1 << int(np.ceil(np.log2(len(x) + L)))
    wet = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(ir, n), n)[: len(x)]
    return x * (1 - mix) + wet * mix * 2.0


# соль-диез минор: E – B – G#m – F#
CHORDS = [
    (40, [56, 59, 64, 68]),   # E
    (35, [54, 59, 63, 66]),   # B
    (44, [56, 59, 63, 68]),   # G#m
    (42, [54, 58, 61, 66]),   # F#
]

keys = np.zeros(N)
pads = np.zeros(N)
bass = np.zeros(N)
voc = np.zeros(N)
nb = int(np.ceil(DUR / BAR)) + 1
for b in range(nb):
    t0 = b * BAR - 0.9          # музыка «уже идёт» на первом кадре
    root, ch = CHORDS[b % 4]
    place(pads, pad(ch[:3], BAR), t0)
    place(keys, piano(ch[0] - 12, BAR * .9, .55, .5), t0)
    for i, nt in enumerate(ch):
        place(keys, piano(nt, BAR * .9, .42, .6), t0 + i * 0.02)
    for i, nt in enumerate([ch[1] + 12, ch[2] + 12, ch[3] + 12, ch[2] + 12, ch[1] + 12, ch[3]]):
        place(keys, piano(nt, BEAT * .7, .30, .9), t0 + (i + 2) * BEAT / 2)
    L = int(BAR * SR)
    tt = np.arange(L) / SR
    place(bass, np.sin(2 * np.pi * midi(root) * tt) * np.minimum(1, tt / .05) * np.exp(-tt * .5) * 0.35, t0)

# две «вокальные» фразы — громкостные пики референса (≈6–8.5 c и ≈14–16.5 c)
place(voc, lead([(75, 0, .5), (73, .5, .5), (71, 1.0, .7), (68, 1.7, 1.3)]), 5.9)
place(voc, lead([(71, 0, .4), (73, .4, .4), (75, .8, .6), (78, 1.4, .6), (75, 2.0, 1.2)]), 13.9)

music = keys * 1.1 + pads * 0.8 + bass * 0.25 + voc * 1.2
music = fft_filter(music, lo=70, hi=6500, order=1)     # «slowed» глуховатость
music = reverb(music, seed=3)

t = np.arange(N) / SR
music *= np.clip(t / 0.35, 0, 1)
# лёгкий подъём к наезду камеры
music *= 1 + 0.25 * np.clip((t - T['zoom']) / (T['glitch'] - T['zoom']), 0, 1)

# громкость как у референса: около -21 dBFS
seg = music[int(1 * SR): int(18 * SR)]
music *= 10 ** (-21 / 20) / (np.sqrt((seg ** 2).mean()) + 1e-12)

# ---------- глитч: заикание, битовое дробление, шумовые всплески ----------
g0, g1 = int(T['glitch'] * SR), int(T['black'] * SR)
gl = music[g0:g1].copy()
chunk = int(0.05 * SR)
src = music[g0 - chunk * 4:g0].copy()
for k in range(0, len(gl), chunk):
    step = k // chunk
    piece = src[(step % 4) * chunk:(step % 4 + 1) * chunk] if step % 3 != 2 else music[g0 + k:g0 + k + chunk]
    piece = piece[: len(gl) - k]
    gl[k:k + len(piece)] = piece
q = 2 ** 4
gl = np.round(gl * q * 3) / (q * 3)                       # дробление по амплитуде
hold = 6
gl = np.repeat(gl[::hold], hold)[: g1 - g0]                # понижение частоты дискретизации
noise = fft_filter(rng.normal(0, 1, g1 - g0), lo=1800, hi=9000) * 0.09
gate = (np.floor(np.arange(g1 - g0) / chunk) % 2 == 0) * 0.9 + 0.1
music[g0:g1] = gl * 1.1 + noise * gate
music[g1:] = 0

# ---------- тихие звуки интерфейса ----------
sfx = np.zeros(N)


def pop(f0, f1, dur=0.12):
    L = int(dur * SR)
    tt = np.arange(L) / SR
    return np.sin(2 * np.pi * np.cumsum(f0 + (f1 - f0) * tt / dur) / SR) * np.exp(-tt * 30) * np.minimum(1, tt / 0.003)


for m in tl['msgs']:
    if m['t'] > 0:
        place(sfx, pop(900, 1500) if m['side'] == 'out' else pop(1200, 800), m['t'], 0.05)
place(sfx, pop(600, 1100, .15), T['rec'], 0.05)
place(sfx, pop(900, 450, .15), T['recCancel'], 0.05)
sfx[g1:] = 0

out = music + sfx
st = np.stack([out, np.roll(out, int(0.0005 * SR))], 1)
st = np.tanh(st / 0.9) * 0.9
with wave.open(OUT, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((st * 32767).astype(np.int16).tobytes())
for s in range(0, int(DUR) + 1):
    sg = st[s * SR:(s + 1) * SR, 0]
    if len(sg):
        print(f"{s:3d}s {20 * np.log10(np.sqrt((sg ** 2).mean()) + 1e-9):6.1f} dB")
