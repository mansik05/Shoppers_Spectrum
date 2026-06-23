import streamlit as st
import pandas as pd
import joblib


# ----------------------
# LOAD MODELS
# ----------------------

kmeans = joblib.load(
    "models/kmeans_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

similarity_df = joblib.load(
    "models/product_similarity.pkl"
)

cluster_labels = joblib.load(
    "models/cluster_labels.pkl"
)


# --------------------
# PAGE
# --------------------

st.set_page_config(
    page_title="Shopper Spectrum",
    layout="wide"
)

st.title(
    "Shopper Spectrum"
)

st.write(
    "Customer Segmentation and Product Recommendation"
)


# --------------------
# PRODUCT RECOMMENDER
# --------------------

st.header(
    "Product Recommendation"
)

product_name = st.text_input(
    "Enter Product Name"
)

if st.button(
    "Get Recommendations"
):

    if product_name in similarity_df.columns:

        recommendations = (
            similarity_df[
                product_name
            ]
            .sort_values(
                ascending=False
            )
            .iloc[1:6]
        )

        st.success(
            "Top 5 Recommended Products"
        )

        for item in recommendations.index:

            st.write(
                "",
                item
            )

    else:

        st.error(
            "Product Not Found"
        )


# ---------------------
# CUSTOMER SEGMENT
# ---------------------

st.header(
    "Customer Segmentation"
)

recency = st.number_input(
    "Recency",
    min_value=0
)

frequency = st.number_input(
    "Frequency",
    min_value=0
)

monetary = st.number_input(
    "Monetary",
    min_value=0.0
)

if st.button(
    "Predict Cluster"
):

    data = scaler.transform(
        [[
            recency,
            frequency,
            monetary
        ]]
    )

    cluster = kmeans.predict(
        data
    )[0]

    segment = cluster_labels[
        cluster
    ]

    st.success(
        f"Customer Segment: {segment}"
    )