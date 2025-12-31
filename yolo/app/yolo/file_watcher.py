"""
파일 감시 모듈
app/yolo 폴더에 새 이미지 파일이 추가되면 자동으로 얼굴 디텍션을 수행합니다.
"""

import os
import time
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# yolo_detection 모듈 import
try:
    from .yolo_detection import detect_faces
except ImportError:
    # 직접 실행 시
    from yolo_detection import detect_faces


class ImageFileHandler(FileSystemEventHandler):
    """이미지 파일 생성/수정 이벤트 핸들러"""
    
    def __init__(self, watch_dir: str, processed_files: set = None):
        """
        Args:
            watch_dir: 감시할 디렉토리 경로
            processed_files: 이미 처리한 파일 목록 (중복 처리 방지)
        """
        self.watch_dir = Path(watch_dir)
        self.processed_files = processed_files if processed_files is not None else set()
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        
    def is_image_file(self, file_path: Path) -> bool:
        """파일이 이미지 파일인지 확인"""
        return file_path.is_file() and file_path.suffix.lower() in self.image_extensions
    
    def should_process(self, file_path: Path) -> bool:
        """파일을 처리해야 하는지 확인"""
        # 이미지 파일이고, 아직 처리하지 않은 파일인지 확인
        if not self.is_image_file(file_path):
            return False
        
        # -detected 접미사가 있는 파일은 제외 (결과 파일)
        if "-detected" in file_path.stem:
            return False
        
        # 이미 처리한 파일인지 확인
        file_key = str(file_path.resolve())
        if file_key in self.processed_files:
            return False
        
        return True
    
    def on_created(self, event):
        """파일 생성 이벤트 처리"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # 파일이 완전히 쓰여질 때까지 대기
        time.sleep(0.5)
        
        if self.should_process(file_path):
            self.process_image(file_path)
    
    def on_modified(self, event):
        """파일 수정 이벤트 처리"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # 파일이 완전히 쓰여질 때까지 대기
        time.sleep(0.5)
        
        if self.should_process(file_path):
            self.process_image(file_path)
    
    def process_image(self, file_path: Path):
        """이미지 파일에 대해 얼굴 디텍션 수행"""
        file_key = str(file_path.resolve())
        
        try:
            print(f"\n{'=' * 60}")
            print(f"새 이미지 파일 감지: {file_path.name}")
            print(f"전체 경로: {file_path}")
            print(f"{'=' * 60}")
            
            # 얼굴 디텍션 수행
            result = detect_faces(str(file_path), save_result=True)
            
            if result["success"]:
                face_count = result["total_objects"]
                result_path = result.get("result_image_path")
                
                print(f"\n{'=' * 60}")
                print(f"얼굴 디텍션 완료!")
                print(f"감지된 얼굴 수: {face_count}개")
                if result_path:
                    print(f"결과 이미지 저장 위치: {result_path}")
                print(f"{'=' * 60}\n")
                
                # 처리 완료된 파일로 표시
                self.processed_files.add(file_key)
            else:
                error_msg = result.get("error", "알 수 없는 오류")
                print(f"\n오류 발생: {error_msg}\n")
                
        except Exception as e:
            print(f"\n이미지 처리 중 오류 발생: {e}\n")
            import traceback
            traceback.print_exc()


def scan_existing_images(watch_dir: str, processed_files: set):
    """기존 이미지 파일들을 스캔하여 처리"""
    watch_path = Path(watch_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    
    print(f"\n기존 이미지 파일 스캔 중...")
    
    for file_path in watch_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            # -detected 접미사가 있는 파일은 제외
            if "-detected" not in file_path.stem:
                file_key = str(file_path.resolve())
                if file_key not in processed_files:
                    print(f"기존 이미지 발견: {file_path.name}")
                    # 처리하지 않고 processed_files에만 추가 (이미 처리된 것으로 간주)
                    processed_files.add(file_key)
    
    print(f"기존 이미지 스캔 완료\n")


def watch_yolo_folder(watch_dir: str = None, scan_existing: bool = True):
    """
    yolo 폴더를 감시하여 새 이미지 파일이 추가되면 자동으로 얼굴 디텍션 수행
    
    Args:
        watch_dir: 감시할 디렉토리 경로 (None이면 기본 경로 사용)
        scan_existing: 기존 이미지 파일도 스캔할지 여부
    """
    # 스크립트 파일의 디렉토리 기준으로 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if watch_dir is None:
        # 기본 경로: app/yolo 폴더
        watch_dir = script_dir
    
    watch_path = Path(watch_dir).resolve()
    
    if not watch_path.exists():
        print(f"오류: 감시할 디렉토리가 존재하지 않습니다: {watch_path}")
        return
    
    print(f"{'=' * 60}")
    print(f"파일 감시 시작")
    print(f"감시 디렉토리: {watch_path}")
    print(f"{'=' * 60}\n")
    
    # 이벤트 핸들러 생성
    processed_files = set()
    event_handler = ImageFileHandler(str(watch_path), processed_files)
    
    # 기존 이미지 스캔 (선택사항)
    if scan_existing:
        scan_existing_images(str(watch_path), processed_files)
    
    # Observer 생성 및 시작
    observer = Observer()
    observer.schedule(event_handler, str(watch_path), recursive=False)
    observer.start()
    
    try:
        print("파일 감시 중... (Ctrl+C로 종료)")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n파일 감시를 종료합니다...")
        observer.stop()
    
    observer.join()
    print("파일 감시가 종료되었습니다.")


if __name__ == "__main__":
    watch_yolo_folder()

