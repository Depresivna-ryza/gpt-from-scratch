
import os
import random

LEIPZIG_URL_100K = "https://downloads.wortschatz-leipzig.de/corpora/eng_news_2025_100K.tar.gz"
LEIPZIG_URL_1M = "https://downloads.wortschatz-leipzig.de/corpora/eng_news_2025_1M.tar.gz"
DEVEL_URL = "https://is.muni.cz/el/fi/jaro2026/PV026/um/data/devel.tsv"
EVAL_URL = "https://is.muni.cz/el/fi/jaro2026/PV026/um/data/eval-input.tsv"


DATA_DIR = "data/datasets"
LEIPZIG_SENTENCES = "eng_news_2025_1M-sentences.txt"
DEVEL_TSV = "devel.tsv"
EVAL_TSV = "eval-input.tsv"


VAL_SPLIT = 0.05
SHUFFLE_SEED = None

def download_datasets():
    import requests
    import tarfile

    # Download and extract Leipzig sentences
    print(f"Downloading Leipzig sentences from {LEIPZIG_URL_1M}...")
    response = requests.get(LEIPZIG_URL_1M, stream=True)
    response.raise_for_status()
    
    with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
        tar.extractall(path=DATA_DIR)
        extracted_directory = os.path.join(DATA_DIR, "eng_news_2025_1M")
        
        if os.path.exists(extracted_directory):
            for filename in os.listdir(extracted_directory):
                if filename.endswith("-sentences.txt"):
                    os.rename(os.path.join(extracted_directory, filename), os.path.join(DATA_DIR, LEIPZIG_SENTENCES))
                    break
        else:
            raise FileNotFoundError(f"Expected extracted directory not found: {extracted_directory}")
        
        # remove the directory after extracting the needed file and also remove the files
        if os.path.exists(extracted_directory):
            for filename in os.listdir(extracted_directory):
                os.remove(os.path.join(extracted_directory, filename))
            os.rmdir(extracted_directory)
        
    print("Leipzig sentences downloaded and extracted successfully!")


    # Download devel.tsv
    print(f"Downloading devel.tsv from {DEVEL_URL}...")
    response = requests.get(DEVEL_URL)
    response.raise_for_status()
    with open(os.path.join(DATA_DIR, DEVEL_TSV), "wb") as f:
        f.write(response.content)
    print("devel.tsv downloaded successfully!")

    # Download eval.tsv
    print(f"Downloading eval.tsv from {EVAL_URL}...")
    response = requests.get(EVAL_URL)
    response.raise_for_status()
    with open(os.path.join(DATA_DIR, EVAL_TSV), "wb") as f:
        f.write(response.content)
    print("eval.tsv downloaded successfully!")


def create_directory_structure():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory at: {DATA_DIR}")
    else:
        print(f"Data directory already exists at: {DATA_DIR}")

def validate_training_data():
    leipzig_full_path = os.path.join(DATA_DIR, LEIPZIG_SENTENCES)
    devel_full_path = os.path.join(DATA_DIR, DEVEL_TSV)
    eval_full_path = os.path.join(DATA_DIR, EVAL_TSV)
    
    if not os.path.exists(leipzig_full_path):
        raise FileNotFoundError(f"Could not find Leipzig sentences file at: {leipzig_full_path}")
    
    print("Leipzig corpus detected successfully!")
    
    # Quick sanity check on the first line to see if it contains IDs
    with open(leipzig_full_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        print(f"Sample raw line: {repr(first_line)}")
        
    if not os.path.exists(devel_full_path):
        raise FileNotFoundError(f"Could not find devel.tsv file at: {devel_full_path}")
    
    print("Devel TSV file detected successfully!")

    with open(devel_full_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        part1, part2 = first_line.strip().split("\t")
        print(f"Sample raw line: {repr(first_line)}")
        print(f"Sample part 1 (grammatical): {repr(part1)}")
        print(f"Sample part 2 (ungrammatical): {repr(part2)}")
        
    if not os.path.exists(eval_full_path):
        raise FileNotFoundError(f"Could not find eval.tsv file at: {eval_full_path}")
    
    print("Eval TSV file detected successfully!")
    
    with open(eval_full_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        part1, part2 = first_line.strip().split("\t")
        print(f"Sample raw line: {repr(first_line)}")
        print(f"Sample part 1 : {repr(part1)}")
        print(f"Sample part 2 : {repr(part2)}")


def _load_all_sentences():
    """
    Load and combine sentences from both source files with balanced weighting:
      - Leipzig: tab-separated ID + sentence, take the sentence part
      - devel.tsv: tab-separated correct + incorrect, take the correct (first) sentence
 
    Devel sentences are oversampled (repeated + randomly drawn) to match the
    Leipzig count so each source contributes ~50% of every batch.
 
    Returns a shuffled list of all sentences.
    """
    # Leipzig corpus
    leipzig_full_path = os.path.join(DATA_DIR, LEIPZIG_SENTENCES)
    if not os.path.exists(leipzig_full_path):
        raise FileNotFoundError(f"Could not find Leipzig sentences file at: {leipzig_full_path}")
    leipzig_sentences = []
    with open(leipzig_full_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            sentence = parts[1] if len(parts) > 1 else parts[0]
            if sentence:
                leipzig_sentences.append(sentence)
 
    # devel.tsv — grammatical (first-column) sentences only
    devel_full_path = os.path.join(DATA_DIR, DEVEL_TSV)
    if not os.path.exists(devel_full_path):
        raise FileNotFoundError(f"Could not find devel.tsv file at: {devel_full_path}")
    devel_sentences = []
    with open(devel_full_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                devel_sentences.append(parts[0])
 
    # Oversample devel to match the Leipzig count so both sources are equally
    # represented in the shuffled pool (~50% each per batch on average).
    rng = random.Random(SHUFFLE_SEED)
    target = len(leipzig_sentences)
    # full_repeats, remainder = divmod(target, len(devel_sentences))
    full_repeats = 15
    devel_oversampled = devel_sentences * full_repeats
 
    print(
        f"Dataset sizes — Leipzig: {len(leipzig_sentences)}, "
        f"Devel (raw): {len(devel_sentences)}, "
        f"Devel (oversampled): {len(devel_oversampled)}"
    )
 
    combined = leipzig_sentences + devel_oversampled
    rng.shuffle(combined)
    return combined


def parquets_iter_batched(split, batch_size=128):
    """
    Iterate over sentences in batches for the given split.
 
    Both Leipzig and devel.tsv sentences are merged into a single pool,
    shuffled with a fixed seed, then split into train / val by VAL_SPLIT.
 
    Args:
        split: "train" or "val"
        batch_size: number of sentences per yielded batch
    """
    all_sentences = _load_all_sentences()
 
    split_idx = int(len(all_sentences) * (1 - VAL_SPLIT))
    if split == "train":
        sentences = all_sentences[:split_idx]
    elif split == "val":
        sentences = all_sentences[split_idx:]
    else:
        raise ValueError(f"Unknown split '{split}'. Expected 'train' or 'val'.")
 
    for i in range(0, len(sentences), batch_size):
        yield sentences[i : i + batch_size]


if __name__ == "__main__":
    # for split in ["train", "val"]:
    #     print(f"Testing {split} split:")
    #     for batch in parquets_iter_batched(split):
    #         print(f"Batch of {len(batch)} sentences, sample: {repr(batch[0])}")
    #         break
    
    create_directory_structure()
    download_datasets()
    validate_training_data()

    # Load raw sentences to build a lookup set for devel
    devel_sentences_raw = set()
    with open(os.path.join(DATA_DIR, DEVEL_TSV), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                devel_sentences_raw.add(parts[0])
 
    CHECK_BATCHES = 20
    total_seen = devel_seen = 0
 
    print(f"\nChecking first {CHECK_BATCHES} train batches for devel representation...")
    for i, batch in enumerate(parquets_iter_batched("train")):
        if i >= CHECK_BATCHES:
            break
        for s in batch:
            print(s)
            total_seen += 1
            if s in devel_sentences_raw:
                devel_seen += 1
 
    leipzig_seen = total_seen - devel_seen
    print(f"Sentences inspected : {total_seen}")
    print(f"  From Leipzig       : {leipzig_seen} ({100 * leipzig_seen / total_seen:.1f}%)")
    print(f"  From devel         : {devel_seen}   ({100 * devel_seen / total_seen:.1f}%)")
 
    # Also show val split size
    val_sentences = list(s for batch in parquets_iter_batched("val") for s in batch)
    print(f"\nVal split size: {len(val_sentences)} sentences")
    print(f"Sample val sentence: {repr(val_sentences[0])}")
    
    
    