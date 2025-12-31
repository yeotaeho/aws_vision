import os
from pathlib import Path

# 프로젝트 루트: diffusers/
BASE_DIR = Path(__file__).resolve().parents[2]
# diffusion 모듈 루트: app/diffusion/
DIFFUSION_DIR = Path(__file__).resolve().parents[1]

OUTPUTS_DIR = DIFFUSION_DIR / "outputs"
IMAGES_DIR = OUTPUTS_DIR / "images"
META_DIR = OUTPUTS_DIR / "metadata"

# 모델 캐시 위치(원하면 바꾸세요)
HF_CACHE_DIR = Path(os.getenv("HF_HOME", str(BASE_DIR / ".hf_cache")))

# RTX 3050 6GB 기본 추천: SDXL Turbo
MODEL_ID = os.getenv("MODEL_ID", "stabilityai/sdxl-turbo")

# 디바이스/정밀도
DEVICE = os.getenv("DEVICE", "cuda")  # cuda / cpu
DTYPE = os.getenv("DTYPE", "float16")  # float16 권장 (cuda에서)

# 6GB 안전 기본값 (Turbo 기준)
DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "768"))
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "768"))
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "4"))
DEFAULT_GUIDANCE = float(os.getenv("DEFAULT_GUIDANCE", "0.0"))

# OOM 방지 상한 (Turbo/6GB 기준 보수적으로)
MAX_WIDTH = int(os.getenv("MAX_WIDTH", "768"))
MAX_HEIGHT = int(os.getenv("MAX_HEIGHT", "768"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))

# 동시성 제한(6GB는 1이 운영적으로 안전)
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))

# URL prefix (리버스프록시/도메인 붙이면 사용)
PUBLIC_IMAGE_BASE = os.getenv("PUBLIC_IMAGE_BASE", "/outputs/images")
PUBLIC_META_BASE = os.getenv("PUBLIC_META_BASE", "/outputs/metadata")