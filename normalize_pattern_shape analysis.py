import os
import mido
import numpy as np
from collections import Counter, defaultdict

# =========================
# 設定
# =========================
MIDI_FOLDER = "midis"
N_GRAM = 5
TOP_K = 20
ROUND_DIGITS = 3
# =========================


# =========================
# キープロファイル
# =========================
MAJOR_PROFILE = np.array([
6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88
])

MINOR_PROFILE = np.array([
6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17
])


# =========================
# 同時音を1音に統合
# =========================
def extract_mono_notes(track):

    time_notes = defaultdict(list)
    t = 0

    for msg in track:
        t += msg.time

        if msg.type == "note_on" and msg.velocity > 0:
            time_notes[t].append(msg.note)

    mono = []

    for t in sorted(time_notes):
        # 最高音 = 主旋律近似
        mono.append(max(time_notes[t]))

    return mono


# =========================
# キー推定
# =========================
def detect_key(notes):

    hist = np.zeros(12)

    for n in notes:
        hist[n % 12] += 1

    scores = []

    for i in range(12):
        maj = np.corrcoef(hist, np.roll(MAJOR_PROFILE, i))[0,1]
        min_ = np.corrcoef(hist, np.roll(MINOR_PROFILE, i))[0,1]

        scores.append((maj, i))
        scores.append((min_, i))

    best = max(scores, key=lambda x: x[0])

    return best[1]


# =========================
# トニック正規化
# =========================
def normalize_notes(notes, tonic):
    return [(n - tonic) % 12 for n in notes]


# =========================
# 度数差列
# =========================
def intervals(notes):
    return [notes[i+1] - notes[i] for i in range(len(notes)-1)]


# =========================
# n-gram生成
# =========================
def ngrams(seq, n):
    return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]


# =========================
# ★ 相対位相正規化（核心）
# =========================
def normalize_pattern_shape(pattern):

    base = pattern[0]
    shifted = [x - base for x in pattern]

    maxv = max(abs(x) for x in shifted)

    if maxv == 0:
        return tuple([0]*len(pattern))

    return tuple(round(x/maxv, ROUND_DIGITS) for x in shifted)


# =========================
# トラック解析
# =========================
def analyze_track(track):

    notes = extract_mono_notes(track)

    if len(notes) < 5:
        return None

    tonic = detect_key(notes)

    norm_notes = normalize_notes(notes, tonic)

    ivals = intervals(norm_notes)

    raw_patterns = ngrams(ivals, N_GRAM)

    # 相対位相化
    normalized_patterns = [
        normalize_pattern_shape(p)
        for p in raw_patterns
    ]

    return normalized_patterns


# =========================
# メイン解析
# =========================
def main():

    global_counter = Counter()

    files = [f for f in os.listdir(MIDI_FOLDER) if f.endswith(".mid")]

    for file in files:

        print(f"\n====== {file} ======")

        path = os.path.join(MIDI_FOLDER, file)
        midi = mido.MidiFile(path)

        for ti, track in enumerate(midi.tracks):

            patterns = analyze_track(track)

            if not patterns:
                continue

            counter = Counter(patterns)
            global_counter.update(counter)

            print(f"\n-- Track {ti} Top Patterns --")

            for pat, count in counter.most_common(TOP_K):
                print(pat, count)


    print("\n================ GLOBAL =================")

    for pat, count in global_counter.most_common(TOP_K):
        print(pat, count)


# =========================
# 実行
# =========================
if __name__ == "__main__":
    main()
