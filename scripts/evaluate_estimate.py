OUT_COMPARISSON_FILE = "data/datasets/eval_comparisson.tsv"
SKIP = 50


with open("data/datasets/eval-input.tsv", "r") as f:
    lines = f.readlines()
    
with open("data/datasets/eval_output.txt", "r") as f:
    outputs = f.readlines()
    
    
for (i, (line, output)) in enumerate(zip(lines, outputs)):
    in_1, in_2 = line.strip().split("\t")
    out = output.strip()

    
    if out != in_1 and out != in_2:
        print(f"Warning: output does not match either input for line {i+1}")
        print(f"Input 1: {in_1}")
        print(f"Input 2: {in_2}")
        print(f"Output : {out}")
        print("-" * 40)
        continue
    else:
        # print(f"Line {i+1}: output matches input {'1' if out == in_1 else '2'}")
        pass
    
    
    if i % SKIP != 0:
        continue
    
    
    with open(OUT_COMPARISSON_FILE, "a") as f:
        f.write(f"{in_1}\t{in_2}\t{out}\n")
    
    