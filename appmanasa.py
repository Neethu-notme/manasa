import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page Settings
st.set_page_config(page_title="Customer Segmentation", page_icon="📊", layout="wide")

# Custom Design (CSS)
st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.3rem;
    }
    .sub-title {
        color: #FFB6C1;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }
    .card {
        background: #FF69B4;
        border: 1px solid #2a2f3a;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 14px;
    }
    .small-note {
        color: #9aa0a6;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Load Model files
std = joblib.load("model_file/scaler.pkl")
pca = joblib.load("model_file/pca.pkl")
kmeans = joblib.load("model_file/kmeans.pkl")

with open("model_file/features.json", "r") as f:
    scl_features = json.load(f)

# Feature Engineering
def feature_engineering(df):
    df = df.copy()
    # Create TotalSpend
    df["TotalSpend"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
        + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )
    # Create TotalPurchases
    df["TotalPurchases"] = (
        df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
    )
    return df

st.markdown('<div class="main-title">📌 Customer Segmentation Deployment (KMeans)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload customer data → Predict clusters → View distribution & PCA plot → Download output</div>', unsafe_allow_html=True)

st.sidebar.title("⚙️ Controls")

st.title("Customer Segmentation Deployment (KMeans)")
st.write("Upload Customer data and get predicted customer clusters..")

upload_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])

fill_missing = st.sidebar.checkbox("Auto Fill Missing Values", value=True)
show_pca_plot = st.sidebar.checkbox("Show PCA Plot", value=True)
show_download = st.sidebar.checkbox("Enable Download Output", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("✅ *Model:* KMeans")
st.sidebar.markdown(f"✅ *Features Used:* {scl_features}")
st.sidebar.markdown("📌 *Output:* Cluster label for each customer")

if upload_file is None:
    st.markdown("""
    <div class="card">
        <b>📂 Upload your dataset from the sidebar.</b><br>
        <span class="small-note">Supported formats: CSV, XLSX</span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if upload_file.name.endswith(".csv"):
    df = pd.read_csv(upload_file)
else:
    df = pd.read_excel(upload_file)

st.markdown('<div class="card"><b>✅ Uploaded Data Preview</b></div>', unsafe_allow_html=True)
st.dataframe(df.head(10), use_container_width=True)
if st.button("Predict Clusters"):
    df = feature_engineering(df)
    
    x = df[scl_features].copy() 

    if fill_missing:
        for col in scl_features:
            x[col] = x[col].fillna(x[col].median())
    else:
        x = x.dropna()

    valid_idx = x.index
    x_scaled = std.transform(x)
    x_pca = pca.transform(x_scaled)
    labels = kmeans.predict(x_pca)

    df["Predicted_Cluster"] = np.nan
    df.loc[valid_idx, "Predicted_Cluster"] = labels
    df["Predicted_Cluster"] = df["Predicted_Cluster"].astype("Int64")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Clustered Customers", len(valid_idx))
    col3.metric("Unclustered (Missing Data)", len(df) - len(valid_idx))

    st.markdown('<div class="card"><b>✅ Output with Predicted Clusters</b></div>', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown('<div class="card"><b>📊 Cluster Distribution</b></div>', unsafe_allow_html=True)
    cluster_counts = df["Predicted_Cluster"].value_counts(dropna=False)
    st.write(cluster_counts)

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    cluster_counts.dropna().sort_index().plot(kind="bar", ax=ax1)
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Customer Count")
    ax1.set_title("Cluster Distribution")
    st.pyplot(fig1)

    if show_pca_plot:
        st.markdown("""
        <div class="card">
            <b>📌 PCA Cluster Plot</b><br>
            <span class="small-note">This plot shows how customers are grouped in 2D after PCA reduction.</span>
        </div>
        """, unsafe_allow_html=True)

        pca_df = pd.DataFrame(x_pca, columns=["PC1", "PC2"])
        pca_df["Cluster"] = labels

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette="tab10", ax=ax2)
        ax2.set_title("KMeans Clusters (PCA View)")
        st.pyplot(fig2)
    if show_download:
        st.markdown("""
        <div class="card">
            <b>⬇️ Download Output</b><br>
            <span class="small-note">Download the dataset with Predicted_Cluster column.</span>
        </div>
        """, unsafe_allow_html=True)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Clustered CSV",
            data=csv_data,
            file_name="customer_clusters_output.csv",
            mime="text/csv"
        )