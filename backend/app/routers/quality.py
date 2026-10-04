import io

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.agents.quality_agent import get_quality_agent, get_surface_quality_agent
from app.schemas import QualityInspectionResponse, SurfaceInspectionResponse

router = APIRouter(prefix="/quality", tags=["Manufacturing Quality Agent"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_PIXELS = 40_000_000               # ~40 MP decoded, well above any real inspection image
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}


def _read_validated_image(file: UploadFile) -> bytes:
    """Read the upload with a size cap and confirm it really is an image.

    PIL is used only to VERIFY here -- a small PNG can declare enormous
    dimensions and expand to hundreds of MB on decode. The actual decode for
    inference goes through TensorFlow so preprocessing matches training; see
    quality_agent.to_model_input.
    """
    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported content type {file.content_type!r}; "
                   f"expected one of {sorted(ALLOWED_TYPES)}")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit")
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")

    try:
        probe = Image.open(io.BytesIO(data))
        width, height = probe.size
        probe.verify()   # structural check without decoding pixel data
    except Image.DecompressionBombError as e:
        # PIL raises this from Image.open itself once a header declares more
        # than 2x MAX_IMAGE_PIXELS (~179 MP), which is BELOW our own MAX_PIXELS
        # check on the next line -- so the guard written for exactly this attack
        # was unreachable for the largest bombs, and a 91-byte PNG produced a
        # 500. DecompressionBombError inherits straight from Exception, so the
        # tuple below never caught it. Same answer as our own cap: 413.
        raise HTTPException(
            status_code=413,
            detail=f"image dimensions exceed the {MAX_PIXELS // 1_000_000} MP decode limit") from e
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise HTTPException(status_code=400, detail="file is not a readable image") from e

    if width * height > MAX_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"image is {width}x{height} ({width * height / 1e6:.1f} MP); "
                   f"the limit is {MAX_PIXELS // 1_000_000} MP")
    return data


# NOTE: these are deliberately `def`, not `async def`. They do heavy synchronous
# work (ResNet50 forward pass, a GradientTape backward pass, PNG encode). As
# `async def` they ran directly on the event loop and blocked every other request
# on the server for the duration; as plain `def`, FastAPI runs them in a
# threadpool, which is what every other router here already does.

# An agent that cannot run is a 503 whether the missing piece is a 98 MB
# artifact or the TensorFlow wheel that would load it. See quality_agent's
# import guard.
_UNAVAILABLE = (FileNotFoundError, ImportError)


@router.post("/inspect", response_model=QualityInspectionResponse)
def inspect_component(file: UploadFile = File(...)):
    data = _read_validated_image(file)
    try:
        result = get_quality_agent().predict(data)
    except _UNAVAILABLE as e:
        raise HTTPException(
            status_code=503,
            detail="the casting quality agent is not available on this deployment; "
                   "see /ready") from e
    return QualityInspectionResponse(**result)


@router.post("/inspect-surface", response_model=SurfaceInspectionResponse)
def inspect_surface(file: UploadFile = File(...)):
    data = _read_validated_image(file)
    try:
        result = get_surface_quality_agent().predict(data)
    except _UNAVAILABLE as e:
        raise HTTPException(
            status_code=503,
            detail="the surface quality agent is not available on this deployment; "
                   "see /ready") from e
    return SurfaceInspectionResponse(**result)
