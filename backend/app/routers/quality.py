from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
from app.schemas import QualityInspectionResponse, SurfaceInspectionResponse
from app.agents.quality_agent import get_quality_agent, get_surface_quality_agent

router = APIRouter(prefix="/quality", tags=["Manufacturing Quality Agent"])


@router.post("/inspect", response_model=QualityInspectionResponse)
async def inspect_component(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        agent = get_quality_agent()
        result = agent.predict(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return QualityInspectionResponse(**result)


@router.post("/inspect-surface", response_model=SurfaceInspectionResponse)
async def inspect_surface(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        agent = get_surface_quality_agent()
        result = agent.predict(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return SurfaceInspectionResponse(**result)
