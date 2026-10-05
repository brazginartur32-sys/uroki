"""Синтез звуковой дорожки с нуля (без чужих сэмплов):
грустный lo-fi бит с пианино -> «провал» в разреженное пианино с реверберацией,
плюс щелчки клавиатуры, звук отправки и входящих сообщений.

python3 audio.py timeline.json audio.wav
"""
import json
import sys
import wave

import numpy as np

SR = 44100
rng = np.random.default_rng(12)

tl = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
DUR = tl['duration']
DROP = tl['drop']
N = int(DUR * SR) + SR

BPM = 75
BEAT = 60 / BPM            # 0.8 c
BAR = 4 * BEAT             # 3.2 c


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def place(buf, sig, t, gain=1.0):
    i = int(t * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i] * gain


def fft_filter(x, lo=None, hi=None, order=2):
    """Мягкий фильтр в частотной области (без scipy)."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    H = np.ones_like(f)
    if hi:
        H *= 1 / np.sqrt(1 + (f / hi) ** (2 * order))
    if lo:
        H *= 1 / np.sqrt(1 + (lo / np.maximum(f, 1e-3)) ** (2 * order))
    return np.fft.irfft(X * H, len(x))


# ---------------- инструменты ----------------
def piano(note, dur, vel=0.8, bright=1.0):
    f0 = midi(note)
    L = int((dur + 1.6) * SR)
    t = np.arange(L) / SR
    out = np.zeros(L)
    B = 0.00035
    for k in range(1, 11):
        fk = k * f0 * np.sqrt(1 + B * k * k)
        if fk > 9000:
            break
        amp = (1 / k ** 1.25) * (bright if k > 2 else 1)
        dec = 0.9 + 0.55 * k + f0 / 900
        for det in (-0.6, 0.6):  # две «струны»
            out += amp * 0.5 * np.sin(2 * np.pi * fk * (1 + det / 1730) * t + rng.random() * 6.28) * np.exp(-t * dec)
    att = np.minimum(1, t / 0.004)
    rel = np.clip(1 - (t - dur) / 0.35, 0, 1)
    hammer = rng.normal(0, 1, L) * np.exp(-t * 180) * 0.08
    return (out * att * rel + hammer * rel) * vel * 0.32


def bass(note, dur, vel=0.8):
    f0 = midi(note)
    L = int((dur + 0.3) * SR)
    t = np.arange(L) / SR
    s = np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(4 * np.pi * f0 * t) + 0.08 * np.sin(6 * np.pi * f0 * t)
    env = np.minimum(1, t / 0.02) * np.exp(-t * 0.6) * np.clip(1 - (t - dur) / 0.25, 0, 1)
    return s * env * vel * 0.5


def kick():
    L = int(0.45 * SR)
    t = np.arange(L) / SR
    f = 45 + 75 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-t * 7.5) + rng.normal(0, 1, L) * np.exp(-t * 300) * 0.15
    return s * 0.55


def snare():
    L = int(0.35 * SR)
    t = np.arange(L) / SR
    n = fft_filter(rng.normal(0, 1, L), lo=900, hi=6000)
    s = n * np.exp(-t * 16) * 0.55 + np.sin(2 * np.pi * 185 * t) * np.exp(-t * 28) * 0.35
    return s * 0.75


def hat(open_=False):
    L = int((0.25 if open_ else 0.07) * SR)
    t = np.arange(L) / SR
    n = fft_filter(rng.normal(0, 1, L), lo=6500, hi=13000)
    return n * np.exp(-t * (14 if open_ else 70)) * 0.22


def key_click():
    L = int(0.05 * SR)
    t = np.arange(L) / SR
    n = fft_filter(rng.normal(0, 1, L), lo=1800, hi=7000)
    s = n * np.exp(-t * 160) + np.sin(2 * np.pi * (1300 + rng.random() * 300) * t) * np.exp(-t * 200) * 0.4
    return s


def send_whoosh():
    L = int(0.22 * SR)
    t = np.arange(L) / SR
    n = rng.normal(0, 1, L)
    # «вжух»: шум с быстро нарастающим фильтром
    out = np.zeros(L)
    for a, b, lo, hi in ((0, .07, 600, 2500), (.05, .14, 1500, 5000), (.11, .22, 3000, 8000)):
        seg = (t >= a) & (t < b)
        part = np.zeros(L)
        part[seg] = n[seg]
        out += fft_filter(part, lo=lo, hi=hi)
    env = np.sin(np.pi * np.clip(t / 0.22, 0, 1)) ** 1.5
    tone = np.sin(2 * np.pi * np.cumsum(500 + 1400 * t / 0.22) / SR) * 0.25
    return (out * 0.8 + tone) * env


def recv_pop():
    L = int(0.25 * SR)
    t = np.arange(L) / SR
    s = np.sin(2 * np.pi * np.cumsum(780 + 420 * np.exp(-t * 40)) / SR) * np.exp(-t * 22)
    s += 0.35 * np.sin(2 * np.pi * 1560 * t) * np.exp(-t * 35)
    return s * np.minimum(1, t / 0.003)


def reverb(x, length=2.6, decay=0.55, mix=0.35, seed=0):
    r = np.random.default_rng(seed)
    L = int(length * SR)
    t = np.arange(L) / SR
    ir = r.normal(0, 1, L) * np.exp(-t / decay)
    ir = fft_filter(ir, hi=5000)
    ir[: int(0.012 * SR)] = 0  # предзадержка
    ir /= np.sqrt((ir ** 2).sum())
    n = 1 << int(np.ceil(np.log2(len(x) + L)))
    wet = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(ir, n), n)[: len(x)]
    return x * (1 - mix) + wet * mix * 2.2


# ---------------- музыка ----------------
# Am – F – C – G (A3..E5)
CHORDS = [
    (45, [57, 60, 64, 69]),  # Am
    (41, [57, 60, 65, 69]),  # F
    (48, [55, 60, 64, 67]),  # C
    (43, [55, 59, 62, 67]),  # G
]
# простая грустная мелодия на 4 такта (нота, доля начала, длительность в долях)
MELODY = [
    [(76, 0, 1.5), (74, 1.5, 0.5), (72, 2, 2)],
    [(72, 0, 1), (69, 1, 1), (72, 2, 1.5), (74, 3.5, 0.5)],
    [(76, 0, 2), (79, 2, 1), (76, 3, 1)],
    [(74, 0, 1.5), (72, 1.5, 0.5), (71, 2, 2)],
]
MELODY_B = [
    [(81, 0, 1), (79, 1, 1), (76, 2, 2)],
    [(77, 0, 1.5), (76, 1.5, 0.5), (72, 2, 2)],
    [(76, 0, 1), (79, 1, 1), (84, 2, 1), (83, 3, 1)],
    [(83, 0, 2), (79, 2, 2)],
]

keys_bus = np.zeros(N)
mel_bus = np.zeros(N)
bass_bus = np.zeros(N)
drum_bus = np.zeros(N)
sparse_bus = np.zeros(N)

nbars = int(np.ceil(DUR / BAR)) + 1
for b in range(nbars):
    t0 = b * BAR
    if t0 > DUR:
        break
    root, ch = CHORDS[b % 4]
    if t0 + 0.01 < DROP:
        # арпеджио восьмыми с лёгким свингом
        pattern = [ch[0], ch[2], ch[3], ch[1], ch[2], ch[3], ch[1], ch[2]]
        for i, nt in enumerate(pattern):
            tt = t0 + i * BEAT / 2 + (0.035 if i % 2 else 0)
            if tt >= DROP:
                break
            place(keys_bus, piano(nt, BEAT * 0.9, vel=0.55 + 0.1 * (i % 4 == 0), bright=0.7), tt)
        if b >= 1:
            place(bass_bus, bass(root - 12 + 12 * (root < 43), BAR * 0.95, vel=0.9), t0)
        if b >= 2:
            mel = (MELODY if (b // 4) % 2 == 0 else MELODY_B)[b % 4]
            for nt, st, ln in mel:
                tt = t0 + st * BEAT
                if tt < DROP - 0.1:
                    place(mel_bus, piano(nt, ln * BEAT, vel=0.75, bright=1.0), tt)
        if b >= 1:
            for beat in range(4):
                tb = t0 + beat * BEAT
                if tb >= DROP:
                    break
                if beat in (0,):
                    place(drum_bus, kick(), tb)
                if beat == 1 and True:
                    pass
                if beat in (1, 3):
                    place(drum_bus, snare(), tb)
                if beat == 2:
                    place(drum_bus, kick(), tb + BEAT * 0.5 + 0.035, 0.8)
                for h in range(2):
                    th = tb + h * BEAT / 2 + (0.045 if h else 0)
                    if th < DROP:
                        place(drum_bus, hat(open_=(beat == 3 and h == 1)), th, 0.9 if h == 0 else 0.6)
    else:
        # после «провала»: редкие ноты, много воздуха
        first = t0 < DROP + BAR
        notes = {0: [(69, 0), (76, 2)], 1: [(72, 0), (77, 2.5)], 2: [(72, 0), (79, 2)], 3: [(71, 0), (74, 2)]}[b % 4]
        for nt, st in notes:
            place(sparse_bus, piano(nt, BEAT * 2.2, vel=0.55, bright=0.8), t0 + st * BEAT)
        place(sparse_bus, piano(ch[0] - 12, BAR, vel=0.22, bright=0.5), t0)

# отдельно — аккорд-«удар» прямо в момент провала
_, ch = CHORDS[int(DROP / BAR) % 4]
for nt in ch:
    place(sparse_bus, piano(nt, 2.5, vel=0.4, bright=0.6), DROP)
place(sparse_bus, piano(ch[0] - 24, 3.0, vel=0.25, bright=0.4), DROP)

keys_bus = fft_filter(keys_bus, hi=3200)          # lo-fi глушение
mel_bus = fft_filter(mel_bus, hi=4200)
drum_bus = fft_filter(drum_bus, hi=7500)

music = keys_bus * 0.8 + mel_bus * 0.95 + bass_bus * 0.28 + drum_bus * 0.5
music = reverb(music, length=2.0, decay=0.45, mix=0.22, seed=1)
sparse = reverb(fft_filter(sparse_bus, hi=3800), length=4.0, decay=1.3, mix=0.55, seed=2)

# крэкл пластинки + шипение
crackle = np.zeros(N)
idx = rng.choice(N, size=int(DUR * 14), replace=False)
crackle[idx] = rng.normal(0, 1, len(idx)) * rng.random(len(idx)) ** 3
crackle = fft_filter(crackle, lo=900, hi=6000) * 0.8 + fft_filter(rng.normal(0, 1, N), lo=2000, hi=9000) * 0.004

t = np.arange(N) / SR
# огибающие секций
duck = np.clip((DROP - t) / 0.06, 0, 1)             # резкий обрыв бита
fade_in = np.clip(t / 0.25, 0, 1)
fade_out = np.clip((DUR - t) / 2.2, 0, 1) ** 1.5
music = music * duck
mix_m = (music + sparse * 0.85) * fade_in * fade_out


def norm_rms(x, db, start=None, end=None):
    seg = x[int((start or 0) * SR): int((end or DUR) * SR)]
    r = np.sqrt((seg ** 2).mean()) + 1e-12
    return x * (10 ** (db / 20) / r)


# громкость: до провала около -20 dBFS (как в референсе), после — тише сама по себе
gain = 10 ** (-20 / 20) / (np.sqrt((mix_m[int(4 * SR): int(50 * SR)] ** 2).mean()) + 1e-12)
mix_m *= gain
crackle = norm_rms(crackle, -46) * fade_in * fade_out

# ---------------- эффекты интерфейса ----------------
sfx = np.zeros(N)
sends = {round(m['t'], 3) for m in tl['msgs'] if m['side'] == 'out' and m['t'] > 0}
for k in tl['keys']:
    if round(k, 3) in sends:
        continue
    place(sfx, key_click(), k, 0.16 * (0.8 + 0.4 * rng.random()))
for m in tl['msgs']:
    if m['t'] <= 0:
        continue
    if m['side'] == 'out':
        place(sfx, send_whoosh(), m['t'], 0.16)
    else:
        place(sfx, recv_pop(), m['t'], 0.13)

out = mix_m + crackle + sfx
# стерео: лёгкая ширина за счёт задержки реверба/крэкла
left = out + crackle * 0.3
right = np.roll(out, int(0.0004 * SR)) - crackle * 0.3
st = np.stack([left, right], 1)[: int(DUR * SR)]
# мягкий лимитер: тихие места линейны, пики скругляются
st = np.tanh(st / 0.9) * 0.9
peak = np.abs(st).max()
pcm = (st * 32767).astype(np.int16)
with wave.open(OUT, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())

for s in range(0, int(DUR), 4):
    seg = st[s * SR:(s + 4) * SR, 0]
    print(f"{s:3d}s {20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-9):6.1f} dB")
print('peak', peak)
