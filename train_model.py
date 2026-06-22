import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


print("Loading Dataset... ")

df = pd.read_csv("C:/Users/mansi/Desktop/ML/Shopper Spectrum project/data/online_retail.csv")

# ----------------------------------------------------------
# DATA CLEANING Removing the Cancelled orders from our data
# ----------------------------------------------------------

df = df.dropna(subset=["CustomerID"])

df = df[
    ~df["InvoiceNo"]
    .astype(str)
    .str.startswith("C")
]

df = df[df["Quantity"] > 0]

df = df[df["UnitPrice"] > 0]

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"]
)

df["TotalAmount"] = (
    df["Quantity"]
    * df["UnitPrice"]
)

print("Data Cleaned")


# --------------------------
# RFM ANALYSIS
# --------------------------

snapshot_date = df["InvoiceDate"].max()  #finds the most recent date or snapshot of recent date

rfm = df.groupby(
    "CustomerID"
).agg(
    {
        "InvoiceDate":
        lambda x:
        (
            snapshot_date
            - x.max()       #finds customers absolute recent last purchase
        ).days,             # the less the no. of days the latest the purchase

        "InvoiceNo":"nunique",  #find the unique transactions performed

        "TotalAmount":"sum"     # this calculates the sum for the total amount spent by customerID
    }
)

# The above calculations and raw columns (Invoice Date , Invoice no., Total Amount) are 
# replaced with Recency , Frequency, Monetary i.e rfm

rfm.columns = [
    "Recency",
    "Frequency",
    "Monetary"
]

print("RFM Created")


# --------------------------
# SCALING
# --------------------------

scaler = StandardScaler()         # Normalizing the above rfm vlaues calculated

rfm_scaled = scaler.fit_transform(
    rfm
)

print("Scaling Done")


# --------------------------
# KMEANS
# --------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(
    rfm_scaled
)

print("Clustering Completed")


# --------------------------
# CLUSTER LABELS
# --------------------------

summary = rfm.groupby(
    "Cluster"
)[
    ["Recency",
     "Frequency",
     "Monetary"]
].mean()

print(summary)

cluster_labels = {
    0:"High-Value",
    1:"Regular",
    2:"Occasional",
    3:"At-Risk"
}


# --------------------------
# PRODUCT RECOMMENDATION
# --------------------------

customer_product = pd.pivot_table(
    df,
    index="CustomerID",
    columns="Description",
    values="Quantity",
    fill_value=0
)

product_matrix = customer_product.T

similarity = cosine_similarity(
    product_matrix
)

similarity_df = pd.DataFrame(
    similarity,
    index=product_matrix.index,
    columns=product_matrix.index
)

print("Recommendation Model Ready")


# --------------------------
# SAVE FILES
# --------------------------

joblib.dump(
    kmeans,
    "models/kmeans_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    similarity_df,
    "models/product_similarity.pkl"
)

joblib.dump(
    cluster_labels,
    "models/cluster_labels.pkl"
)

print("Models Saved Successfully")