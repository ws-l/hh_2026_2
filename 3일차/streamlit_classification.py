# app.py

import streamlit as st
import numpy as np
from PIL import Image
import joblib

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="이미지 분류기",
    layout="wide"
)

st.title("32x32 컬러 JPG 불량 판별")
st.write("업로드한 이미지를 Decision Tree 모델로 분류")

# -----------------------------
# 모델 로드
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model1.pkl")   # 저장해둔 DT 모델 파일명
    return model

model = load_model()

# 클래스 이름이 있으면 지정
class_names = {
    0: "class_0",
    1: "class_1",
    2: "class_2",
    3: "class_3",
    4: "class_4",
    5: "class_5",
    6: "class_6",
    7: "class_7",
    8: "class_8"
}

# -----------------------------
# 이미지 전처리 함수
# -----------------------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")  # 컬러 3채널 강제
    img = img.resize((32, 32))                      # 32x32로 맞춤
    img_array = np.array(img, dtype=np.uint8)       # shape: (32, 32, 3)

    # DT 모델 입력용으로 1차원 펼치기
    x = img_array.flatten().reshape(1, -1)          # shape: (1, 3072)
    return img, img_array, x

# -----------------------------
# 사이드바
# -----------------------------
st.sidebar.header("업로드 옵션")
uploaded_file = st.sidebar.file_uploader(
    "JPG 파일 업로드",
    type=["jpg", "jpeg"]
)

show_array = st.sidebar.checkbox("이미지 배열 shape 보기", value=False)

# -----------------------------
# 본문
# -----------------------------
if uploaded_file is not None:
    try:
        img, img_array, x = preprocess_image(uploaded_file)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("업로드 이미지")
            st.image(img, caption="입력 이미지", use_container_width=True)

        with col2:
            st.subheader("예측 결과")

            pred = model.predict(x)[0]           
            pred_class = np.argmax(pred, axis=-1)
            st.success(f"예측 클래스: {class_names[pred_class]}")

            pred2 = model.predict_proba(x)[pred_class][0][1]          
            st.success(f"예측 확률: {pred2*100:.2f}%")
            #if hasattr(model, "predict_proba"):
            #   proba = model.predict_proba(x)[0]
            #    st.write("클래스별 확률")
            #    for i, p in enumerate(proba):
            #        label = class_names.get(i, str(i))
            #        st.write(f"- {label}: {p:.4f}")

        if show_array:
            st.subheader("전처리 정보")
            st.write("원본 배열 shape:", img_array.shape)
            st.write("모델 입력 shape:", x.shape)
            st.write("앞부분 값 예시:", x[0][:20])

    except Exception as e:
        st.error(f"처리 중 오류가 발생했습니다: {e}")

else:
    st.info("왼쪽 사이드바에서 JPG 이미지를 업로드하세요.")