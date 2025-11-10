import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

# Initialize MTCNN once (device auto-select if CUDA is available)
_mtcnn = MTCNN(keep_all=True)

# ArcFace-like canonical 5-point template at 112x112, scaled to 256x256
_ARCFACE_5PTS_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

_SCALE = 256.0 / 112.0
_ARCFACE_5PTS_256 = _ARCFACE_5PTS_112 * _SCALE

def detect_face_5pt(pil_img, prob_thresh=0.90):
    """
    Run MTCNN on a PIL image, return the best face: (box, landmarks, prob)
    - box: [x1,y1,x2,y2]
    - landmarks: 5x2 array (float32) in image coords
    """
    img_rgb = np.array(pil_img)  # RGB
    boxes, probs, landmarks = _mtcnn.detect(img_rgb, landmarks=True)
    if boxes is None or len(boxes) == 0:
        return None, None, None
    # filter by prob
    candidates = []
    for b, p, lm in zip(boxes, probs, landmarks):
        if p is None or p < prob_thresh:
            continue
        x1, y1, x2, y2 = b
        area = max(0, x2-x1) * max(0, y2-y1)
        candidates.append((area, b, lm, p))
    if not candidates:
        return None, None, None
    # pick largest area
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, box, lm, prob = candidates[0]
    return box.astype(np.float32), lm.astype(np.float32), float(prob)

def estimate_similarity_transform(src_pts, dst_pts):
    """
    Estimate 2x3 similarity (affine partial) transform from src_pts->dst_pts.
    src_pts, dst_pts: Nx2 float32 arrays. N>=3 is okay; N=5 better.
    """
    # OpenCV's estimateAffinePartial2D returns 2x3 matrix
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    return M  # shape (2,3)

def align_face_256(pil_img, landmarks_5pt):
    """
    Warp input image to 256x256 aligned face using 5pt similarity transform.
    Returns aligned PIL image (RGB) and the 2x3 transform matrix.
    """
    src = landmarks_5pt.astype(np.float32)
    dst = _ARCFACE_5PTS_256
    M = estimate_similarity_transform(src, dst)
    if M is None:
        return None, None
    img_rgb = np.array(pil_img)
    aligned = cv2.warpAffine(img_rgb, M, (256, 256), flags=cv2.INTER_LINEAR)
    aligned_pil = Image.fromarray(aligned)  # remains RGB
    return aligned_pil, M
