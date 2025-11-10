import json
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split

def build_identity_splits(aligned_root, test_size=0.1, val_size=0.1, seed=42):
    """
    Split identities into train/val/test disjoint groups. Return a dict:
    {"train": [id1, id2, ...], "val": [...], "test": [...]}
    """
    rng = seed
    ids = []
    id2count = {}
    for person in os.listdir(aligned_root):
        pdir = os.path.join(aligned_root, person)
        if not os.path.isdir(pdir): continue
        cnt = sum(1 for f in os.listdir(pdir) if f.lower().endswith(('.jpg','.jpeg','.png')))
        if cnt > 0:
            ids.append(person)
            id2count[person] = cnt

    # we can stratify by "bucketed counts"
    def bucketize(c):
        if c < 3: return "low"
        if c < 10: return "mid"
        return "high"
    strat = [bucketize(id2count[i]) for i in ids]

    # first split out test
    ids_trainval, ids_test = train_test_split(ids, test_size=test_size, random_state=rng, stratify=strat)
    strat_tv = [bucketize(id2count[i]) for i in ids_trainval]
    # then split train/val
    val_ratio = val_size / (1.0 - test_size)
    ids_train, ids_val = train_test_split(ids_trainval, test_size=val_ratio, random_state=rng, stratify=strat_tv)

    return {"train": ids_train, "val": ids_val, "test": ids_test}

def save_splits_json(splits, out_path):
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
