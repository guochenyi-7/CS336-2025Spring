import os
import string
import unicodedata
import mmh3
import shutil
import random
import networkx as nx
from collections import defaultdict

def normalize_text(text: str)-> str:
    # NFD Unicode 标准化
    text = unicodedata.normalize("NFD", text)

    # 去重音
    text = "".join([c for c in text if unicodedata.category(c) != "Mn"])

    # 转小写
    text = text.lower()

    # 移除标点符号
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # 移除空白
    text = " ".join(text.split())

    return text

def get_ngrams(text: str, ngrams: int)-> set:
    tokens = text.split()

    if len(tokens) < ngrams:
        return set()
    
    n_grams = set()
    for i in range(len(tokens) - ngrams + 1):
        ngrams_tokens = tokens[i: i + ngrams]
        ngrams_str = " ".join(ngrams_tokens)
        n_grams.add(ngrams_str)
    
    return n_grams     

def get_minhash_signature(n_grams, num_hashes):
    signature = [float('inf')] * num_hashes

    for ngrams_tokens in n_grams:
        ngrams_bytes = ngrams_tokens.encode("utf-8")
        for i in range(num_hashes):
            hash_val = mmh3.hash(ngrams_bytes, seed=i)
            if hash_val < signature[i]:
                signature[i] = hash_val

    return signature

def get_candidate_pairs(buckets):
    candidate_pairs = set()
    for _, doc_ids in buckets.items():
        if len(doc_ids) > 1:
            sorted_ids = sorted(doc_ids)
            for i in range(len(sorted_ids)):
                for j in range(i + 1, len(sorted_ids)):
                    pair = (sorted_ids[i], sorted_ids[j])
                    candidate_pairs.add(pair)

    return candidate_pairs

def get_real_jaccard(file_path_a, file_path_b, ngrams_n):
    with open(file_path_a, "r", encoding="utf-8") as f:
        ngrams_a = get_ngrams(normalize_text(f.read()), ngrams_n)

    with open(file_path_b, "r", encoding="utf-8") as f:
        ngrams_b = get_ngrams(normalize_text(f.read()), ngrams_n)

    intersection = len(ngrams_a.intersection(ngrams_b))
    union = len(ngrams_a.union(ngrams_b))

    if union == 0:
        return 0.0
    
    return intersection / union

def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    buckets = defaultdict(list)
    for i, filepath in enumerate(input_files):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            clean_text = normalize_text(text)
            n_grams = get_ngrams(clean_text, ngrams)
            signature = get_minhash_signature(n_grams, num_hashes)

            rows_per_band = num_hashes // num_bands
            for band_idx in range(num_bands):
                start = band_idx * rows_per_band
                end = start + rows_per_band

                band_seg = tuple(signature[start:end])
                buckets[(band_idx, band_seg)].append(i)

    candidate_pairs = get_candidate_pairs(buckets)

    G = nx.Graph()
    G.add_nodes_from(range(len(input_files)))
    for idx_a, idx_b in candidate_pairs:
        file_a = input_files[idx_a]
        file_b = input_files[idx_b]

        real_jaccard = get_real_jaccard(file_a, file_b, ngrams)
        if real_jaccard >= jaccard_threshold:
            G.add_edge(idx_a, idx_b)

    clusters = list(nx.connected_components(G))

    indices_to_remove = set()
    for cluster in clusters:
        cluster_list = list(cluster)
        random.shuffle(cluster_list)
        indices_to_remove.update(cluster_list[1:])

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, filepath in enumerate(input_files):
        if i in indices_to_remove:
            continue
        filename = os.path.basename(filepath)
        final_output_path = os.path.join(output_directory, filename)
        shutil.copy2(filepath, final_output_path)
    