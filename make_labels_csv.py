import csv

# Config
FRAMES_PATH = "frames_test.csv"
INPUTS_PATH = "inputs_test.csv"
OUT_PATH = "labels.csv"

def load_keydowns(inputs_path: str):
    keydowns = []
    with open(inputs_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["event"] == "keydown":
                keydowns.append(float(row["t"]))
    keydowns.sort()
    return keydowns

def load_frames(frames_path: str):
    frames = []
    with open(frames_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append((int(row["frame_idx"]), float(row["t"])))
    frames.sort(key=lambda x: x[0])
    return frames

def make_labels(frames, keydowns, out_path: str):
    k = 0
    n_k = len(keydowns)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "t", "jump"])

        prev_t = 0.0
        for frame_idx, t in frames:
            jump = 0

            # Advance keydowns until in (prev_t, t]
            while k < n_k and keydowns[k] <= t:
                if keydowns[k] > prev_t:
                    jump = 1
                k += 1

            writer.writerow([frame_idx, f"{t:.6f}", jump])
            prev_t = t

if __name__ == "__main__":
    keydowns = load_keydowns(INPUTS_PATH)
    frames = load_frames(FRAMES_PATH)
    make_labels(frames, keydowns, OUT_PATH)
    print(f"Wrote {OUT_PATH} with {len(frames)} rows.")