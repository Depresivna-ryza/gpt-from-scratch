import random

DEVEL_FILE = "data/datasets/devel.tsv"
OUT_VAL_SPLIT_FILE = "data/datasets/devel_val_split.tsv"

VALIDATION_PERCENT = 0.05
RANDOM_SEED = 42

with open(DEVEL_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

random.seed(RANDOM_SEED)
random.shuffle(lines)

split_idx = int(len(lines) * VALIDATION_PERCENT)

val_lines = lines[:split_idx]
train_lines = lines[split_idx:]

# overwrite original with training data
with open(DEVEL_FILE, "w", encoding="utf-8") as f:
    f.writelines(train_lines)

# save validation set
with open(OUT_VAL_SPLIT_FILE, "w", encoding="utf-8") as f:
    f.writelines(val_lines)