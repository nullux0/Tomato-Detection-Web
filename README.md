# 🍅 웹 기반 토마토 숙도 탐지 시스템
## 1.개요
프로젝트 소개본 프로젝트는 과수원에서의 토마토 객체 탐지 시스템입니다.
YOLOv8 모델을 기반으로 학습을 진행하였으며, ONNX 형식으로 변환하여 추론 속도를 최적화했습니다.
(이곳에 데모 실행 화면 캡처 이미지를 1~2장 드래그 앤 드롭으로 추가하세요. 깃허브 웹 화면에서 수정 모드로 들어가 이미지를 끌어다 놓으면 자동으로 코드가 생성됩니다.)

## 2. 개발 환경 및 의존성 
- OS: Windows / Mac / Linux (Cross-platform)
- Language: Python 3.9+
- Framework: Ultralytics (YOLOv8),Streamlit
- 주요 패키지: requirements.txt 참조

## 3. 상세 설치 및 실행 방법
### 3-1. 
모델 가중치 파일 다운로드GitHub 용량 제한(100MB)으로 인해 학습된 모델 파일은 외부 클라우드에 업로드하였습니다. 아래 링크에서 모델을 다운로드하여 프로젝트 최상위 폴더(app.py와 같은 위치)에 넣어주세요.

[🔗 YOLOv8 ONNX 모델 가중치 다운로드 링크] : https://drive.google.com/file/d/1a5u2VST4r9XnzAPbX3bekNC_FiFmD9CX/view?usp=drive_link 
### 3-2. 
로컬 환경 실행 방법터미널을 열고 아래 명령어를 순서대로 실행합니다.

#### 1. 저장소 클론 (다운로드)
    git clone [https://github.com/내아이디/Crop-Detection-Web.git](https://github.com/내아이디/Crop-Detection-Web.git)
    cd Crop-Detection-Web

#### 2. 필요 패키지 설치
    pip install -r requirements.txt

#### 3. Streamlit 웹 서버 실행
    streamlit run app.py

(참고: 코드 내에 하드코딩된 절대 경로(C:\...)가 없으므로 어떤 PC에서든 즉시 구동됩니다.)

## 4. 데이터 파이프라인 
1. 데이터 수집: Roboflow Universe에서 다품종 과일(사과, 토마토, 포도) 이미지 및 Bounding Box 데이터 약 2,000장 확보
2. 전처리: 모든 이미지를 640x640 해상도로 Auto-Orient 및 Resize 처리
3. 모델 학습: Google Colab (T4 GPU) 환경에서 YOLOv8n 모델을 활용하여 Transfer Learning 진행
4. 최적화: 추론 지연 시간(Latency) 최소화를 위해 PyTorch 모델(.pt)을 ONNX 런타임 환경으로 변환
5. 서빙: Streamlit을 활용하여 사용자가 이미지를 업로드하면 즉시 탐지 결과를 시각화해주는 Web UI 구축


## 5. 팀원 역할 분담
[공소영]:
- 커스텀 데이터셋 수집 (Roboflow)
- YOLOv8 모델 Fine-Tuning 및 평가 (Colab)
- 모델 ONNX 변환 및 최적화
- Streamlit 기반 웹 데모 UI/UX 구현 및 깃허브 배포

[정예린]:
- streamlit 기반 UI 구현


