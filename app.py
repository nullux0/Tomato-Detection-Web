import streamlit as st
import cv2
import numpy as np
from PIL import Image
import traceback

from inference import load_model, predict
from utils import draw_boxes

# ==========================================
# 1. 페이지 기본 설정
# ==========================================
st.set_page_config(page_title="토마토 숙도 탐지기", page_icon="🍅", layout="wide")

# ==========================================
# 2. 모델 로드 (@st.cache_resource)
# ==========================================
model = load_model("best.onnx")

# ==========================================
# 3. 사이드바 UI (인터랙티브 제어)
# ==========================================
st.sidebar.header("⚙️ 설정 (Settings)")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.00, 
    max_value=1.00, 
    value=0.40, 
    step=0.05,
    help="오분류(FP)를 줄이려면 임계값을 높이세요" 
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** 슬라이더를 조절하면 화면 새로고침 없이 즉각적으로 박스가 업데이트됩니다.")

# ==========================================
# 4. 메인 화면 UI
# ==========================================
st.title("🍅 실시간 토마토 숙도 탐지 시스템")
st.markdown("과수원 환경에서 수확이 임박한 토마토를 탐지하고 **숙도(fully_ripe, half_ripe, unripe)**를 분류합니다.")

uploaded_file = st.file_uploader("테스트할 토마토 이미지를 업로드하세요", type=['jpg', 'jpeg', 'png'])

# ==========================================
# 5. 핵심 로직 및 예외 처리
# ==========================================
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        boxes, confidences, class_ids = predict(model, image_bgr, conf_threshold)
        
        result_image_bgr = draw_boxes(image_bgr, boxes, class_ids, confidences)
        result_image_rgb = cv2.cvtColor(result_image_bgr, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("원본 이미지")
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("탐지 결과")
            st.image(result_image_rgb, use_container_width=True)
            
            # --- 🚀 추가된 숙도별 카운팅 로직 ---
            st.markdown("### 📊 숙도별 탐지 현황")
            
            # class_ids 배열을 순회하며 숙도별로 개수를 셉니다.
            ripe_count = sum(1 for c in class_ids if int(c) == 0)
            half_count = sum(1 for c in class_ids if int(c) == 1)
            unripe_count = sum(1 for c in class_ids if int(c) == 2)
            
            # Streamlit의 metric 위젯을 사용하여 대시보드 형태로 예쁘게 출력합니다.
            m1, m2, m3 = st.columns(3)
            m1.metric("🔴 완숙 (fully_ripe)", f"{ripe_count}개")
            m2.metric("🟠 반숙 (half_ripe)", f"{half_count}개")
            m3.metric("🟢 미숙 (unripe)", f"{unripe_count}개")
            
            st.success(f"✅ 설정한 임계값({conf_threshold:.2f}) 기준으로 총 **{len(boxes)}개**의 토마토가 탐지되었습니다!")
            
    except Exception as e:
        st.error(f"🚨 에러가 발생했습니다: {e}")
        st.code(traceback.format_exc())