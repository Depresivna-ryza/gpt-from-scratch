with open("data/datasets/devel.tsv") as f:
    lengths = [len(line.split("\t")[0].split()) for line in f]
print(f"Max tokens (words): {max(lengths)}, p95: {sorted(lengths)[int(0.95*len(lengths))]}")