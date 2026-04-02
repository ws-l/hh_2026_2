import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.title("CSV 업로드 후 K-means 군집화")

# 1. CSV 업로드
uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("원본 데이터")
    st.dataframe(df)

    # 2. 수치형 컬럼만 추출
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("군집화를 하려면 수치형 컬럼이 2개 이상 필요합니다.")
    else:
        # 3. 사용할 컬럼 선택
        selected_cols = st.multiselect(
            "군집화에 사용할 컬럼 선택",
            numeric_cols,
            default=numeric_cols
        )

        if len(selected_cols) == 0:
            st.warning("컬럼을 1개 이상 선택하세요.")
        else:
            # 4. k값 슬라이더
            max_k = min(10, len(df))
            k = st.slider("군집 개수(k)", min_value=2, max_value=max_k, value=3)

            # 5. 결측 제거
            X = df[selected_cols].dropna().copy()

            if len(X) < k:
                st.warning("결측 제거 후 데이터 수가 k보다 작습니다. k를 줄이세요.")
            else:
                # 6. 표준화
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # 7. KMeans 수행
                model = KMeans(n_clusters=k, random_state=42, n_init="auto")
                model.fit(X_scaled)

                # 8. 결과 저장
                df["cluster"] = model.labels_

                st.subheader("군집 결과")
                st.dataframe(df)

                # 9. 2개 컬럼 선택 시 산점도
                if len(selected_cols) >= 2:
                    x_col = selected_cols[0]
                    y_col = selected_cols[1]

                    fig, ax = plt.subplots()
                    scatter = ax.scatter(
                        df[x_col],
                        df[y_col],
                        c=df["cluster"]
                    )
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    st.pyplot(fig)

                # 10. CSV 다운로드
                csv = df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="군집 결과 CSV 다운로드",
                    data=csv,
                    file_name="cluster_result.csv",
                    mime="text/csv"
                )