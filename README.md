Shopper Spectrum is an E-Commerce Analytics project that uses RFM Analysis, K-Means Clustering, and Collaborative Filtering to segment customers and recommend products based on purchase behavior.

Objectives of this project:
Segment customers using Recency, Frequency, and Monetary (RFM) analysis.
Identify High-Value, Regular, Occasional, and At-Risk customers.
Build an Item-Based Collaborative Filtering recommendation system.
Deploy the solution using Streamlit.

Technologies Used in this project are:
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
Streamlit
Plotly
Joblib


Machine Learning Models:
Customer Segmentation
RFM Feature Engineering
StandardScaler
K-Means Clustering
Elbow Method & Silhouette Score for evaluation
Product Recommendation
Item-Based Collaborative Filtering
Cosine Similarity

Project Structure of this project:

Shopper-Spectrum/
│
├── data/
│   └── online_retail.csv
│
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── product_similarity.pkl
│   └── cluster_labels.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md

Installation:

Install dependencies:
- pip install -r requirements.txt

Train the models:
- python train_model.py

Run the Streamlit application:
- streamlit run app.py

Features in this project:
- Product Recommendation
- Enter a product name
- Get Top 5 similar product recommendations

Customer Segmentation:
- Input Recency, Frequency, and Monetary values
- Predict customer segment instantly

Business Impact of this project:
- Personalized marketing campaigns
- Customer retention strategies
- Product recommendation and cross-selling
- Better customer understanding through segmentation
