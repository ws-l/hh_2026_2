import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

st.title("Pattern")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    cols = df.columns.tolist()

    id_col = st.selectbox("choose ID", cols)
    item_col = st.selectbox("choose item", cols)

    min_support = st.slider("choose min support", 0.01, 1.00, 0.10, 0.01)
    min_confidence = st.slider("choose min confidence", 0.01, 1.00, 0.50, 0.01)

    if st.button("Find pattern"):
        # 필요한 컬럼만 사용
        data = df[[id_col, item_col]].dropna().copy()

        # 문자열 정리
        data[item_col] = data[item_col].astype(str).str.strip()

        # 주문별-상품별 교차표 생성
        basket = pd.crosstab(data[id_col], data[item_col])

        # 0/1 변환
        basket = (basket > 0).astype(int)

        st.subheader("장바구니 형태 데이터")
        st.dataframe(basket.head())

        # frequent itemsets
        freq_items = apriori(basket, min_support=min_support, use_colnames=True)
        freq_items["length"] = freq_items["itemsets"].apply(len)
        freq_items["itemsets"] = freq_items["itemsets"].apply(
            lambda x: ", ".join(sorted(list(x)))
        )

        st.subheader("Frequent Itemsets")
        if len(freq_items) > 0:
            st.dataframe(freq_items.sort_values("support", ascending=False))
        else:
            st.warning("no result.")

        # rules
        freq_for_rules = apriori(basket, min_support=min_support, use_colnames=True)

        if len(freq_for_rules) > 0:
            rules = association_rules(
                freq_for_rules,
                metric="confidence",
                min_threshold=min_confidence
            )

            if len(rules) > 0:
                rules["antecedents"] = rules["antecedents"].apply(
                    lambda x: ", ".join(sorted(list(x)))
                )
                rules["consequents"] = rules["consequents"].apply(
                    lambda x: ", ".join(sorted(list(x)))
                )

                show_cols = [
                    "antecedents", "consequents",
                    "support", "confidence", "lift"
                ]

                st.subheader("Association Rules")
                st.dataframe(
                    rules[show_cols].sort_values(
                        ["lift", "confidence"], ascending=False
                    )
                )
            else:
                st.info("No pattern")