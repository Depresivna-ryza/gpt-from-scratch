DEVEL_FILE = "data/datasets/devel.tsv"

VALIDATION_SPLIT = 20

OUT_TRAIN_SPLIT_FILE = "data/datasets/devel_train_split.tsv"
OUT_VAL_SPLIT_FILE = "data/datasets/devel_val_split.tsv"


with open(DEVEL_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()
    
    for i, line in enumerate(lines):
        if i % VALIDATION_SPLIT == 0:
            with open(OUT_VAL_SPLIT_FILE, "a", encoding="utf-8") as val_f:
                val_f.write(line)
        else:
            with open(OUT_TRAIN_SPLIT_FILE, "a", encoding="utf-8") as train_f:
                train_f.write(line)