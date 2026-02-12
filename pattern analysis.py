import os
import mido
import numpy as np
from collections import Counter, defaultdict

# ========= 設定 =========
MIDI_FOLDER = "midis"
N_GRAM = 3
TOP_K = 20
# ========================


MAJOR_PROFILE = np.array([
6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88
])

MINOR_PROFILE = np.array([
6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17
])


# ---------- 同時音統合 ----------
def extract_mono_notes(track):

    time_notes = defaultdict(list)
    t = 0

    for msg in track:
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            time_notes[t].append(msg.note)

    mono = []
    for t in sorted(time_notes):
        mono.append(max(time_notes[t]))  # 最高音採用

    return mono


# ---------- キー推定 ----------
def detect_key(notes):
    hist = np.zeros(12)
    for n in notes:
        hist[n % 12] += 1

    scores = []
    for i in range(12):
        maj = np.corrcoef(hist, np.roll(MAJOR_PROFILE, i))[0,1]
        min_ = np.corrcoef(hist, np.roll(MINOR_PROFILE, i))[0,1]
        scores.append((maj, i, "major"))
        scores.append((min_, i, "minor"))

    best = max(scores, key=lambda x: x[0])
    return best[1], best[2]


# ---------- 正規化 ----------
def normalize(notes, tonic):
    return [(n - tonic) % 12 for n in notes]


# ---------- 度数差 ----------
def intervals(notes):
    return [notes[i+1] - notes[i] for i in range(len(notes)-1)]


# ---------- ngram ----------
def ngrams(seq, n):
    return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]


# ---------- 解析 ----------
def analyze_track(track):

    notes = extract_mono_notes(track)
    if len(notes) < 5:
        return None

    tonic, mode = detect_key(notes)
    norm = normalize(notes, tonic)
    ivals = intervals(norm)
    return ngrams(ivals, N_GRAM)


# ---------- メイン ----------
global_patterns = Counter()

files = [f for f in os.listdir(MIDI_FOLDER) if f.endswith(".mid")]

for file in files:
    print(f"\n====== {file} ======")

    mid = mido.MidiFile(os.path.join(MIDI_FOLDER, file))

    for i, track in enumerate(mid.tracks):

        patterns = analyze_track(track)
        if not patterns:
            continue

        counter = Counter(patterns)
        global_patterns.update(counter)

        print(f"\n-- Track {i} Top Patterns --")
        for pat, c in counter.most_common(TOP_K):
            print(pat, c)


print("\n================ GLOBAL =================")
for pat, c in global_patterns.most_common(TOP_K):
    print(pat, c)
