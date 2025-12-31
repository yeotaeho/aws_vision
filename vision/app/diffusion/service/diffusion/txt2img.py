import math
import torch
from app.diffusion.service.diffusion.pipeline_manager import get_pipeline
from app.diffusion.core.config import (
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_STEPS, DEFAULT_GUIDANCE,
    MAX_WIDTH, MAX_HEIGHT, MAX_STEPS, DEVICE
)

def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))

def _round_to_multiple(v: int, base: int = 8) -> int:
    return int(math.floor(v / base) * base)

def generate_txt2img(
    prompt: str,
    negative_prompt: str | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE,
    seed: int | None = None,
):
    pipe = get_pipeline()

    width = _round_to_multiple(_clamp_int(width, 64, MAX_WIDTH), 8)
    height = _round_to_multiple(_clamp_int(height, 64, MAX_HEIGHT), 8)
    steps = _clamp_int(steps, 1, MAX_STEPS)

    gen = None
    if seed is not None:
        device = "cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
        gen = torch.Generator(device=device).manual_seed(int(seed))

    # Turbo는 guidance가 0~1 수준에서 흔히 사용되며,
    # 기본값을 0으로 두는 편이 안정적입니다.
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=float(guidance_scale),
        generator=gen,
    )

    image = result.images[0]

    meta = {
        "model_id": getattr(pipe, "name_or_path", None),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": float(guidance_scale),
        "seed": seed,
        "device": "cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu",
    }
    return image, meta