import os
import random
import csv
from collections import defaultdict
from glob import glob

def list_images_by_id(aligned_root):
    id2imgs = defaultdict(list)
    for person in os.listdir(aligned_root):
        pdir = os.path.join(aligned_root, person)
        if not os.path.isdir(pdir): continue
        imgs = sorted(glob(os.path.join(pdir, "*.*")))
        imgs = [p for p in imgs if p.lower().endswith(('.jpg','.jpeg','.png'))]
        if imgs:
            id2imgs[person] = imgs
    return id2imgs

def make_pairs(aligned_root, splits, split_name, same_ratio=0.5, max_pairs=200000, seed=1337):
    """
    Build pairs for a given split (train/val/test).
    - same_ratio: fraction of same-identity pairs vs cross-identity pairs.
    """
    rnd = random.Random(seed)
    id2imgs = list_images_by_id(aligned_root)
    ids = splits[split_name]
    ids = [i for i in ids if i in id2imgs and len(id2imgs[i])>=1]
    # flatten imgs per split
    all_imgs = [(pid, p) for pid in ids for p in id2imgs[pid]]

    pairs = []
    same_target = int(max_pairs * same_ratio)
    diff_target = max_pairs - same_target

    # same-identity pairs
    cnt_same = 0
    for pid in ids:
        imgs = id2imgs[pid]
        if len(imgs) < 2: continue
        # sample multiple pairs
        sampled = 0
        while sampled < len(imgs) and cnt_same < same_target:
            a, b = rnd.sample(imgs, 2)
            pairs.append((a, b, 1))
            cnt_same += 1
            sampled += 1
        if cnt_same >= same_target: break

    # cross-identity pairs
    cnt_diff = 0
    while cnt_diff < diff_target:
        (pid_a, a) = rnd.choice(all_imgs)
        (pid_b, b) = rnd.choice(all_imgs)
        if pid_a == pid_b: 
            continue
        pairs.append((a, b, 0))
        cnt_diff += 1

    return pairs

def save_pairs_csv(pairs, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_path", "tgt_path", "same_identity"])
        w.writerows(pairs)
