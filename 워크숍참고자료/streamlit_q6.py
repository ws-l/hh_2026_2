# streamlit_autorefresh 설치 필요
import requests
import pandas as pd
import streamlit as st
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import joblib

# 설정
API_URL = "http://127.0.0.1:8000/data?row=1"
REFRESH_SEC = 10


st.set_page_config(page_title="불량 모니터링", layout="wide")
st.title("실시간 불량 모니터링")

# 자동 새로고침
st_autorefresh(interval=REFRESH_SEC * 1000, key="refresh")

# 세션 상태
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=["time", "pred"]
    )

# 모델 로드
@st.cache_resource
def load_model():
    return joblib.load("gbm_model_q6.pkl")

model = load_model()

# API 호출
def fetch():
    try:
        r = requests.get(API_URL, timeout=5)
        return r.json(), None
    except Exception as e:
        return None, str(e)

data, err = fetch()

if err:
    st.error(f"API 오류: {err}")
else:
    # 입력 데이터
    X = pd.DataFrame(data)
    st.dataframe(X)

    # 예측
    pred = model.predict(X)
    pred2 = "defect" if pred == 1 else "normal"  
    now = datetime.now().strftime("%H:%M:%S")

    # 기록 저장
    new_row = pd.DataFrame([{
        "time": now,
        "pred": pred2,
    }])

    st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)

    # 출력 (텍스트 중심)
    st.subheader("현재 상태")
    st.write(now)
    st.write(pred2)
    st.subheader("최근 로그")
    st.dataframe(st.session_state.history)

# 사이드바
with st.sidebar:
    st.write("갱신 주기", REFRESH_SEC)
