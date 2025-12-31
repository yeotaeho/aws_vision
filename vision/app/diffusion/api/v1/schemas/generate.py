from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: str | None = None

    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = None

class GenerateResponse(BaseModel):
    id: str
    image_url: str
    meta_url: str
    meta: dict