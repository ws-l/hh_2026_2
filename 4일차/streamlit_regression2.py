import time
import requests
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="실시간 예측", layout="wide")
st.title("API 기반 실시간 예측 관리도")


# pkl 업로드
uploaded_model = st.file_uploader("pkl 모형 업로드", type=["pkl"])

if uploaded_model is not None:
    try:
        loaded = joblib.load(uploaded_model)
        model = loaded["model"]
        st.success("모형 로드 완료")
        API_URL = "http://127.0.0.1:8000/data_num"
        feature_cols = loaded["cols"]

        pred_history = []
        step_history = []

        chart_area = st.empty()
        table_area = st.empty()
        status_area = st.empty()

        run = st.button("실행 시작")
        if run:
            for i in range(30):
                try:
                    r = requests.get(API_URL)
                    data = r.json()
                   
                    row_df = pd.DataFrame(data, columns=feature_cols)
                    st.write(f"API 응답 데이터: {data}")
                    time.sleep(3)
                    
                    pred = model.predict(row_df)[0]
                    
                    #Control chart
                    pred_history.append(pred)
                    step_history.append(i + 1)

                    arr = np.array(pred_history)
                    CL = np.mean(arr)
                    sigma = np.std(arr, ddof=1) if len(arr) > 1 else 0
                    UCL = CL + 3 * sigma
                    LCL = CL - 3 * sigma

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(step_history, pred_history, marker='o', label="Prediction")
                    ax.axhline(CL, linestyle='--', label=f"CL={CL:.3f}")
                    ax.axhline(UCL, linestyle='--', label=f"UCL={UCL:.3f}")
                    ax.axhline(LCL, linestyle='--', label=f"LCL={LCL:.3f}")
                    ax.set_title("Prediction Control Chart")
                    ax.set_xlabel("Step")
                    ax.set_ylabel("Predicted Value")
                    ax.legend()
                    ax.grid(True)

                    chart_area.pyplot(fig)

                    result_df = pd.DataFrame({
                        "step": step_history,
                        "prediction": pred_history
                    })
                    table_area.dataframe(result_df, use_container_width=True)

                    status_area.success(f"{i+1}번째 데이터 예측 완료: {pred:.4f}")

                except Exception as e:
                    status_area.error(f"오류 발생: {e}")
                    time.sleep(3)

    except Exception as e:
        st.error(f"모형 로드 또는 예측 중 오류 발생: {e}")

else:
    st.warning("pkl 파일 업로드")




