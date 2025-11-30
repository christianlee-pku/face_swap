from registry import MODELS
from models.unet_face_swap import UNetFaceSwap

if "UNetFaceSwap" not in MODELS._items:  # type: ignore[attr-defined]
    MODELS.register("UNetFaceSwap")(UNetFaceSwap)
