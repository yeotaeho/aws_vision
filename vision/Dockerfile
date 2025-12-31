FROM python:3.11-slim

WORKDIR /api

# 시스템 패키지 업데이트 및 의존성 설치
# OpenCV와 이미지 처리에 필요한 라이브러리 추가
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 출력 버퍼링 비활성화
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# requirements.txt 파일들 먼저 복사 (Docker 레이어 캐싱 최적화)
COPY requirements.txt .
COPY app/diffusion/requirements.txt ./app/diffusion/requirements.txt

# Python 의존성 설치
# Diffusion 서브모듈의 requirements.txt에 FastAPI/uvicorn이 포함되어 있으므로 먼저 설치
RUN pip install --no-cache-dir -r app/diffusion/requirements.txt

# 루트 requirements.txt 설치 (diffusion requirements와 중복되지만 호환성 유지)
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 포트 노출
EXPOSE 9020

# 애플리케이션 실행
# app.diffusion.main:app을 메인 앱으로 사용
CMD ["uvicorn", "app.diffusion.main:app", "--host", "0.0.0.0", "--port", "9020"]

