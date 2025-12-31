from fastapi import APIRouter
from app.diffusion.api.v1.schemas.generate import GenerateRequest, GenerateResponse
from app.diffusion.core.limits import get_semaphore
from app.diffusion.service.diffusion.txt2img import generate_txt2img
from app.diffusion.service.storage.filesystem import save_image_and_meta
from app.diffusion.core.config import (
DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_STEPS, DEFAULT_GUIDANCE
)

# 동시성 제한(세마포어) 걸고 생성 후 저장합니다.

router = APIRouter()

@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
        sem = get_semaphore()
        async with sem:
            image, meta = generate_txt2img(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width or DEFAULT_WIDTH,
            height=req.height or DEFAULT_HEIGHT,
            steps=req.steps or DEFAULT_STEPS,
            guidance_scale=req.guidance_scale if req.guidance_scale is not None else DEFAULT_GUIDANCE,
            seed=req.seed,
            )
        saved = save_image_and_meta(image, meta)
        return saved