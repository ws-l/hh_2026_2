import streamlit as st
import pandas as pd
from river import preprocessing, linear_model, metrics

st.title("Online Learning  with river")

# 파일 읽기
df = pd.read_csv("data.csv")

# 컬럼명 정리
df = df.rename(columns={"Pass.Fail": "Pass_Fail"})

st.write("데이터 미리보기")
st.dataframe(df.head())

# 모델 생성
model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
metric = metrics.Accuracy()

# 진행 버튼
if st.button("온라인 학습 시작"):
    results = []

    for i, row in df.iterrows():
        y = row["Pass_Fail"]
        X = row.drop("Pass_Fail").to_dict()

        # 예측
        y_pred = model.predict_one(X)

        # 평가
        if y_pred is not None:
            metric.update(y, y_pred)

        # 학습
        model.learn_one(X, y)

        results.append({
            "step": i + 1,
            "true": y,
            "pred": y_pred,
            "accuracy_so_far": metric.get()
        })

    result_df = pd.DataFrame(results)

    st.success("학습 완료")
    st.write(f"최종 Accuracy: {metric.get():.4f}")

    st.write("학습 진행 결과")
    st.dataframe(result_df.head(20))

    st.line_chart(result_df.set_index("step")["accuracy_so_far"])