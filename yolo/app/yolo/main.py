"""YOLO FastAPI 애플리케이션 메인 모듈."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import os
from datetime import datetime

from .yolo_detection import detect_faces, auto_detect_faces

app = FastAPI(title="YOLO Detection API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:4000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 업로드 디렉토리 설정
UPLOAD_DIR = Path("/api/app/yolo")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 결과 저장 디렉토리
RESULT_DIR = Path("/api/app/data/yolo")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    """루트 엔드포인트."""
    return {
        "message": "YOLO Detection API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
def health():
    """헬스체크 엔드포인트."""
    return {"status": "healthy"}


@app.post("/api/detect")
async def detect_image(file: UploadFile = File(...)):
    """이미지 업로드 및 얼굴 디텍션.
    
    Args:
        file: 업로드된 이미지 파일
        
    Returns:
        디텍션 결과
    """
    try:
        # 파일 저장
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 얼굴 디텍션 수행
        result = detect_faces(str(file_path), save_result=True)
        
        return {
            "success": result["success"],
            "image_path": str(file_path),
            "detected_objects": result.get("detected_objects", {}),
            "total_objects": result.get("total_objects", 0),
            "result_image_path": result.get("result_image_path"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"디텍션 중 오류 발생: {str(e)}")


@app.post("/api/detect/path")
async def detect_image_path(image_path: str):
    """이미지 경로를 받아서 얼굴 디텍션.
    
    Args:
        image_path: 디텍션할 이미지 파일 경로
        
    Returns:
        디텍션 결과
    """
    try:
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="이미지 파일을 찾을 수 없습니다.")
        
        # 얼굴 디텍션 수행
        success, face_count, result_path = auto_detect_faces(image_path)
        
        return {
            "success": success,
            "image_path": image_path,
            "face_count": face_count,
            "result_image_path": result_path,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"디텍션 중 오류 발생: {str(e)}")


@app.post("/api/upload-yolo")
async def upload_yolo(files: list[UploadFile] = File(..., description="업로드할 이미지 파일들")):
    """여러 이미지 파일을 업로드하고 얼굴 디텍션을 수행.
    
    파일은 app/yolo 폴더에 저장되고, 디텍션 결과는 app/data/yolo 폴더에 저장됩니다.
    
    Args:
        files: 업로드된 이미지 파일 리스트 (FormData에서 'files' 이름으로 전송)
        
    Returns:
        업로드 및 디텍션 결과
    """
    try:
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="파일이 없습니다.")
        
        saved_files = []
        detection_results = []
        
        for file in files:
            # 파일명에 타임스탬프 추가하여 중복 방지
            timestamp = int(datetime.now().timestamp() * 1000)
            original_name = file.filename or "unnamed"
            file_name = f"{timestamp}_{original_name}"
            file_path = UPLOAD_DIR / file_name
            
            # 파일 저장 (app/yolo 폴더에 저장)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            saved_files.append(file_name)
            
            # 얼굴 디텍션 수행 (결과는 app/data/yolo에 저장됨)
            try:
                result = detect_faces(str(file_path.resolve()), save_result=True)
                detection_results.append({
                    "file": file_name,
                    "success": result["success"],
                    "total_objects": result.get("total_objects", 0),
                    "detected_objects": result.get("detected_objects", {}),
                    "result_image_path": result.get("result_image_path"),
                    "error": result.get("error"),
                })
            except Exception as e:
                import traceback
                error_detail = str(e)
                traceback.print_exc()
                detection_results.append({
                    "file": file_name,
                    "success": False,
                    "total_objects": 0,
                    "detected_objects": {},
                    "result_image_path": None,
                    "error": error_detail,
                })
        
        return {
            "success": True,
            "message": f"{len(saved_files)}개의 파일이 저장되고 디텍션되었습니다.",
            "files": saved_files,
            "detections": detection_results,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"파일 업로드 및 디텍션 중 오류 발생: {str(e)}")
