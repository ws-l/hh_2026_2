import streamlit as st
import pandas as pd
import joblib


st.set_page_config(page_title="회귀모형 예측 실습", layout="centered")

st.title("회귀모형 예측 실습")
st.write("model pkl 파일을 업로드하여 예측")

# pkl 업로드
uploaded_model = st.file_uploader("pkl 모형 업로드", type=["pkl"])

if uploaded_model is not None:
    try:
        loaded = joblib.load(uploaded_model)
        model = loaded["model"]
        st.success("모형 로드 완료")

        # 컬럼 입력
        st.subheader("입력 방식 선택")
        mode = st.radio("선택", ["직접 입력", "CSV 업로드"])

        # 직접 입력
        if mode == "직접 입력":
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

        # CSV 업로드
        else:
            st.subheader("CSV 업로드")
            st.write(f"CSV는 다음 변수들을 포함해야 합니다: `{', '.join(loaded['cols'])}`")

            uploaded_csv = st.file_uploader("CSV 파일 업로드", type=["csv"], key="csv")

            if uploaded_csv is not None:
                df = pd.read_csv(uploaded_csv)

                st.write("업로드 데이터")
                st.dataframe(df, use_container_width=True)

                missing_cols = [col for col in loaded['cols'] if col not in df.columns]

                if missing_cols:
                    st.error(f"다음 컬럼이 없습니다: {missing_cols}")
                else:
                    if st.button("일괄 예측"):
                        result_df = df.copy()
                        result_df["prediction"] = model.predict(df[loaded['cols']])

                        st.success("예측 완료")
                        st.dataframe(result_df, use_container_width=True)

                        csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="결과 CSV 다운로드",
                            data=csv_data,
                            file_name="prediction_result.csv",
                            mime="text/csv"
                        )

    except Exception as e:
        st.error(f"모형 로드 또는 예측 중 오류 발생: {e}")

else:
    st.warning("먼저 pkl 파일을 업로드하세요.")