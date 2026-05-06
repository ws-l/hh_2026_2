import streamlit as st
import pandas as pd
import joblib


st.set_page_config(page_title="분류 예측 실습", layout="centered")

st.title("분류 모형 예측 실습")
st.write("model pkl 파일을 업로드하여 예측")

# pkl 업로드
uploaded_model = st.file_uploader("pkl 모형 업로드", type=["pkl"])

if uploaded_model is not None:
    try:
        loaded = joblib.load(uploaded_model)
        model = loaded["model"]
        st.success("모형 로드 완료")

        # 컬럼 입력
        
        st.subheader("직접 입력")

        values = {}
        cols = loaded["cols"]

        for i in cols:
                #with cols[i % len(cols)]:
                    values[i] = st.number_input(f"{i}", value=0.0)

        input_df = pd.DataFrame([values])

        st.write("입력 데이터")
        st.dataframe(input_df, use_container_width=True)

        if st.button("예측하기"):
                pred = model.predict(input_df)[0]
                st.success(f"예측값: {pred:,.4f}")
       

    except Exception as e:
        st.error(f"모형 로드 또는 예측 중 오류 발생: {e}")

else:
    st.warning("먼저 pkl 파일을 업로드하세요.")