from __future__ import annotations

import io
import time
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse

from PIL import Image, ImageOps, UnidentifiedImageError

# Pydantic v2 imports
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project utilities (assumed available from your repo)
from utils.image_io import pil_to_numpy_rgb
from utils.infer_core import InferenceConfig, FaceSwapInference

class Settings(BaseSettings):
    """Service configuration loaded from environment variables or .env."""
    GEN_CKPT: str = Field(default="models/gen_B_best.pt", description="Generator checkpoint path")
    ID_ENC_CKPT: Optional[str] = Field(default=None, description="ID encoder checkpoint path (fixed from Stage A)")
    DEVICE: str = Field(default="cuda", description="Compute device: 'cuda' or 'cpu'")
    MAX_IMAGE_MB: int = Field(default=10, description="Max upload size per image (MB)")
    OUTPUT_DIR: str = Field(default="service_outputs", description="Directory to save outputs if requested")
    ALLOW_ORIGINS: str = Field(default="*", description="CORS allowed origins, comma-separated")
    TOKEN: Optional[str] = Field(default=None, description="Optional bearer token for /warmup and /swap")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()

app = FastAPI(title="FaceSwap Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.ALLOW_ORIGINS.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_engine: Optional[FaceSwapInference] = None
_engine_lock = asyncio.Lock()  # guard lazy initialization and cfg updates


async def _get_engine(
    blend: str,
    feather_ksize: int,
    color_match: bool,
    src_index: Optional[int],
    tgt_index: Optional[int],
    mask_scale_x: float,
    mask_scale_y: float,
) -> FaceSwapInference:
    """
    Lazily create the FaceSwapInference engine and update runtime knobs per request.
    Model weights are loaded once and reused across requests/process.
    """
    global _engine
    async with _engine_lock:
        if _engine is None:
            cfg = InferenceConfig(
                ckpt_path=settings.GEN_CKPT,
                id_enc_ckpt_path=settings.ID_ENC_CKPT,
                device=settings.DEVICE,
                blend_mode=blend,
                feather_ksize=feather_ksize,
                color_match=color_match,
                src_index=src_index,
                tgt_index=tgt_index,
                mask_scale_x=mask_scale_x,
                mask_scale_y=mask_scale_y,
            )
            _engine = FaceSwapInference(cfg)
        else:
            # Update runtime parameters without reloading weights
            _engine.cfg.blend_mode = blend
            _engine.cfg.feather_ksize = feather_ksize
            _engine.cfg.color_match = color_match
            _engine.cfg.src_index = src_index
            _engine.cfg.tgt_index = tgt_index
            _engine.cfg.mask_scale_x = mask_scale_x
            _engine.cfg.mask_scale_y = mask_scale_y
    return _engine

class SwapParams(BaseModel):
    """Form parameters accepted by /swap."""
    blend: str = Field(default="feather", description="Blending mode: 'feather' or 'poisson'")
    feather_ksize: int = Field(default=25, ge=1, le=255, description="Feather kernel size for feather mode")
    color_match: bool = Field(default=False, description="Match lightness in HSV before blending")
    src_index: Optional[int] = Field(default=None, description="Pick Nth detected face in source")
    tgt_index: Optional[int] = Field(default=None, description="Pick Nth detected face in target")
    mask_scale_x: float = Field(default=1.0, gt=0.3, le=2.0, description="Ellipse mask horizontal scale in aligned space")
    mask_scale_y: float = Field(default=1.0, gt=0.3, le=2.0, description="Ellipse mask vertical scale in aligned space")
    save_output: bool = Field(default=False, description="If true, save PNG to OUTPUT_DIR and return JSON path")

    @field_validator("blend")
    @classmethod
    def _blend_ok(cls, v: str) -> str:
        if v not in ("feather", "poisson"):
            raise ValueError("blend must be 'feather' or 'poisson'")
        return v


def _require_token(req: Request) -> None:
    """
    Simple optional bearer auth.
    If TOKEN is set in settings, the request must include header: Authorization: Bearer <TOKEN>
    """
    if not settings.TOKEN:
        return
    auth = req.headers.get("Authorization") or ""
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if auth.split(" ", 1)[1] != settings.TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


def _read_image_file(f: UploadFile, max_mb: int) -> Image.Image:
    """
    Read an uploaded image into a PIL Image (RGB) with size cap and EXIF-safe orientation.
    Raises HTTPException for invalid types or oversized payloads.
    """
    content = f.file.read()
    if len(content) > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image too large (>{max_mb} MB)")
    try:
        # Use EXIF transpose to normalize orientation, then convert to RGB
        im = Image.open(io.BytesIO(content))
        im = ImageOps.exif_transpose(im).convert("RGB")
        return im
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail=f"Invalid image: {f.filename or 'uploaded file'}")


def _save_png(arr, out_dir: Path) -> Path:
    """
    Save a numpy RGB array to disk as PNG in the given directory.
    Returns the output path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"swap_{int(time.time()*1000)}.png"
    Image.fromarray(arr).save(p.as_posix())
    return p

@app.post("/swap")
async def swap(
    req: Request,
    src: UploadFile = File(..., description="Source face (identity donor)"),
    tgt: UploadFile = File(..., description="Target image (face receiver)"),
    blend: str = Form("feather"),
    feather_ksize: int = Form(25),
    color_match: bool = Form(False),
    src_index: Optional[int] = Form(None),
    tgt_index: Optional[int] = Form(None),
    mask_scale_x: float = Form(1.0),
    mask_scale_y: float = Form(1.0),
    save_output: bool = Form(False),
):
    """
    Swap the face in `tgt` with the identity from `src`.
    Returns a PNG stream by default, or a JSON object with a saved file path when save_output=true.
    """
    _require_token(req)

    # Validate and compose parameters
    try:
        params = SwapParams(
            blend=blend,
            feather_ksize=feather_ksize,
            color_match=color_match,
            src_index=src_index,
            tgt_index=tgt_index,
            mask_scale_x=mask_scale_x,
            mask_scale_y=mask_scale_y,
            save_output=save_output,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameters: {e}")

    # Read and decode images (with size cap)
    t0 = time.time()
    im_src = _read_image_file(src, settings.MAX_IMAGE_MB)
    im_tgt = _read_image_file(tgt, settings.MAX_IMAGE_MB)

    # Convert PIL -> numpy RGB
    np_src = pil_to_numpy_rgb(im_src)
    np_tgt = pil_to_numpy_rgb(im_tgt)

    # Get engine (lazy load on first call) and run inference
    engine = await _get_engine(
        blend=params.blend,
        feather_ksize=params.feather_ksize,
        color_match=params.color_match,
        src_index=params.src_index,
        tgt_index=params.tgt_index,
        mask_scale_x=params.mask_scale_x,
        mask_scale_y=params.mask_scale_y,
    )

    try:
        result = engine.swap_images(np_src, np_tgt)
    except RuntimeError as e:
        # Typical runtime issues such as "No face detected"
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # Unexpected server-side errors
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    composite = result["composite"]
    elapsed = time.time() - t0

    # Persist to disk if requested (useful for audits)
    if params.save_output:
        out_path = _save_png(composite, Path(settings.OUTPUT_DIR))
        return JSONResponse(
            {
                "ok": True,
                "path": out_path.as_posix(),
                "elapsed_sec": round(elapsed, 3),
                "debug": result.get("debug", {}),
            }
        )

    # Default: return a PNG stream with diagnostic headers
    buf = io.BytesIO()
    Image.fromarray(composite).save(buf, format="PNG")
    buf.seek(0)
    headers = {
        "X-Elapsed": f"{elapsed:.3f}",
        "X-Blend": params.blend,
        "X-ColorMatch": str(params.color_match),
        "X-MaskScale": f"{params.mask_scale_x},{params.mask_scale_y}",
        "X-Device": settings.DEVICE,
    }
    return StreamingResponse(buf, media_type="image/png", headers=headers)


@app.get("/metrics")
async def metrics():
    """
    Minimal text metrics endpoint.
    Integrate Prometheus/OTEL in production if richer metrics are required.
    """
    loaded = _engine is not None
    return PlainTextResponse(
        "\n".join(
            [
                f"engine_loaded {1 if loaded else 0}",
                f"device{{name=\"{settings.DEVICE}\"}} 1",
                f"max_image_mb {settings.MAX_IMAGE_MB}",
            ]
        )
        + "\n"
    )
