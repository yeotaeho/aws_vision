# 이미지와 메타데이터를 파일로 저장하고, 
# URL을 만들어 반환합니다.
import json
import uuid
from datetime import datetime
from pathlib import Path
from PIL import Image

from app.diffusion.core.config import IMAGES_DIR, META_DIR, PUBLIC_IMAGE_BASE, PUBLIC_META_BASE

def ensure_dirs():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

def save_image_and_meta(image: Image.Image, meta: dict) -> dict:
    ensure_dirs()

    uid = uuid.uuid4().hex
    ts = datetime.utcnow().isoformat() + "Z"

    image_name = f"{uid}.png"
    meta_name = f"{uid}.json"

    image_path = IMAGES_DIR / image_name
    meta_path = META_DIR / meta_name

    image.save(image_path, format="PNG")

    meta_out = {
        "id": uid,
        "created_at": ts,
        **meta,
        "image_file": image_name,
        "meta_file": meta_name,
    }
    meta_path.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "id": uid,
        "image_url": f"{PUBLIC_IMAGE_BASE}/{image_name}",
        "meta_url": f"{PUBLIC_META_BASE}/{meta_name}",
        "meta": meta_out,
    }