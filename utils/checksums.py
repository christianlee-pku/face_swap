import hashlib
import numpy as np
from PIL import Image

def sha256_file(path, chunk_size=1<<20):
    """
    Compute SHA-256 for a file (exact-duplicate detection).
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def dhash_image(pil_img, hash_size=8):
    """
    Compute difference hash (dHash) for near-duplicate detection.
    Returns integer you can compare via Hamming distance.
    """
    # resize to (hash_size+1, hash_size) grayscale
    gray = pil_img.convert("L").resize((hash_size+1, hash_size), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    # pack bits
    val = 0
    for bit in diff.flatten():
        val = (val << 1) | int(bit)
    return val

def hamming_distance(a, b):
    return (a ^ b).bit_count()
