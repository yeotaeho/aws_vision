# FastAPI 엔트리 + 정적 파일 서빙(/outputs/...) 
# + 라우팅 등록입니다.
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# app/diffusion 디렉토리에서 직접 실행할 때 프로젝트 루트를 sys.path에 추가
_current_file = Path(__file__).resolve()
if _current_file.parent.name == "diffusion":
    project_root = _current_file.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.diffusion.api.v1.routes.generate import router as generate_router
from app.diffusion.core.config import OUTPUTS_DIR

app = FastAPI(title="Diffusers API", version="1.0.0")

# CORS 설정 (프론트엔드에서 API 호출 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# outputs 디렉토리가 없으면 생성
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# outputs 정적 서빙 (로컬 개발/단독 서버에서 편리)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

app.include_router(generate_router, prefix="/api/v1")

@app.get("/health")
def health():
    return {"ok": True}