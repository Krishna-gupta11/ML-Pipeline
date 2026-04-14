import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, classification_report

# ---------------- UI CONFIG ----------------
st.set_page_config(layout="wide", page_title="Uber ML Pipeline")

# Custom CSS to fix the "Squashed" text in navigation buttons
st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        border-radius: 4px;
        height: 3.5em;
        background-color: #f8f9fb;
        border: 1px solid #d1d5db;
        color: #1f2937;
        font-weight: 600;
        white-space: normal; /* Allows text to wrap if button is too small */
        line-height: 1.2;
    }
    div.stButton > button:hover {
        border-color: #ff4b4b;
        color: #ff4b4b;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if 'df' not in st.session_state: st.session_state.df = None
if 'model' not in st.session_state: st.session_state.model = None
if 'page' not in st.session_state: st.session_state.page = "Data Input"

# ---------------- HELPER: DISTANCE ----------------
def haversine(lat1, lon1, lat2, lon2):
    r = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# ---------------- SIDEBAR ----------------
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload uber.csv", type=["csv"])

# ---------------- HORIZONTAL NAVIGATION ----------------
nav_list = ["Data Input", "EDA", "Cleaning", "Features", "Split", "Model Selection", "Training", "Metrics", "Predict"]
cols = st.columns(len(nav_list))
for i, name in enumerate(nav_list):
    if cols[i].button(name):
        st.session_state.page = name

st.divider()

if uploaded_file is not None:
    if st.session_state.df is None:
        st.session_state.df = pd.read_csv(uploaded_file)
    
    df = st.session_state.df
    page = st.session_state.page

    # 1. DATA INPUT
    if page == "Data Input":
        st.subheader("📂 Initial Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")

    # 2. EDA
    elif page == "EDA":
        st.subheader("📊 Visual Analysis")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df[df['fare_amount'] < 60], x='fare_amount', title="Fare Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.write("**Feature Correlation**")
            fig_corr, ax = plt.subplots()
            sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='viridis', ax=ax)
            st.pyplot(fig_corr)

    # 3. CLEANING
    elif page == "Cleaning":
        st.subheader("🧹 Data Sanitization")
        if st.button("Drop Missing & Outlier Fares"):
            df = df.dropna()
            df = df[(df['fare_amount'] > 2) & (df['fare_amount'] < 200)]
            st.session_state.df = df
            st.success("Data cleaned!")

    # 4. FEATURES
    elif page == "Features":
        st.subheader("📌 Feature Engineering")
        if st.button("Compute Uber Features"):
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
            df['hour'] = df['pickup_datetime'].dt.hour
            df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
            df['dist_km'] = haversine(df['pickup_latitude'], df['pickup_longitude'], 
                                     df['dropoff_latitude'], df['dropoff_longitude'])
            df = df[(df['dist_km'] > 0) & (df['dist_km'] < 100)]
            st.session_state.df = df
            st.success("Generated 'dist_km', 'hour', and 'day_of_week'!")

    # 5. SPLIT
    elif page == "Split":
        st.subheader("✂️ Train/Test Split")
        tsize = st.slider("Test Set Size %", 10, 50, 20)
        feats = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'hour', 'day_of_week', 'dist_km']
        
        X = df[feats]
        y = df['fare_amount']
        
        st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(X, y, test_size=tsize/100, random_state=42)
        st.success(f"Split Complete. Training rows: {len(st.session_state.X_train)}")

    # 6. MODEL SELECTION
    elif page == "Model Selection":
        st.subheader("🤖 Choose Your Algorithm")
        st.session_state.problem_type = st.radio("Pipeline Mode", ["Regression", "Classification"])
        
        if st.session_state.problem_type == "Regression":
            algo = st.selectbox("Select Regressor", ["Linear Regression", "Random Forest Regressor", "KNN Regressor"])
            if algo == "KNN Regressor":
                k_n = st.slider("Neighbors (n_neighbors)", 1, 20, 5)
                st.session_state.model = KNeighborsRegressor(n_neighbors=k_n)
            elif algo == "Random Forest Regressor":
                st.session_state.model = RandomForestRegressor(n_estimators=50)
            else:
                st.session_state.model = LinearRegression()
        
        else:
            st.info("Classification will predict if the fare is Above Median (High) or Below Median (Low).")
            algo = st.selectbox("Select Classifier", ["Logistic Regression", "Random Forest Classifier", "KNN Classifier", "Naive Bayes"])
            if algo == "KNN Classifier":
                k_n = st.slider("Neighbors (n_neighbors)", 1, 20, 5)
                st.session_state.model = KNeighborsClassifier(n_neighbors=k_n)
            elif algo == "Naive Bayes":
                st.session_state.model = GaussianNB()
            elif algo == "Random Forest Classifier":
                st.session_state.model = RandomForestClassifier(n_estimators=50)
            else:
                st.session_state.model = LogisticRegression()

    # 7. TRAINING
    elif page == "Training":
        st.subheader("🏋️ Training & Validation")
        k_folds = st.number_input("K-Fold Splits", 2, 10, 5)
        
        if st.button("Start Training"):
            with st.spinner("Processing..."):
                # Setup targets for classification if needed
                y_train = st.session_state.y_train
                if st.session_state.problem_type == "Classification":
                    median_val = st.session_state.df['fare_amount'].median()
                    y_train = (y_train > median_val).astype(int)
                
                # Full training
                st.session_state.model.fit(st.session_state.X_train, y_train)
                
                # Cross Validation (on a sample for speed)
                scores = cross_val_score(st.session_state.model, st.session_state.X_train[:2000], y_train[:2000], cv=k_folds)
                st.success("Model trained successfully!")
                st.write(f"Mean K-Fold Accuracy/Score: {scores.mean():.4f}")

    # 8. METRICS
    elif page == "Metrics":
        st.subheader("📈 Performance Reports")
        if st.session_state.model:
            y_pred = st.session_state.model.predict(st.session_state.X_test)
            y_test = st.session_state.y_test
            
            if st.session_state.problem_type == "Regression":
                st.metric("R2 Score", f"{r2_score(y_test, y_pred):.4f}")
                st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
            else:
                y_test_bin = (y_test > st.session_state.df['fare_amount'].median()).astype(int)
                st.metric("Accuracy", f"{accuracy_score(y_test_bin, y_pred):.4f}")
                st.text("Classification Report:")
                st.text(classification_report(y_test_bin, y_pred))
        else:
            st.warning("Train the model first.")

    # 9. PREDICT
    elif page == "Predict":
        st.subheader("🔮 Predictive Query")
        if st.session_state.model:
            with st.form("input_form"):
                col1, col2 = st.columns(2)
                with col1:
                    plat = st.number_input("Pickup Lat", value=40.71)
                    plon = st.number_input("Pickup Lon", value=-74.0)
                with col2:
                    dlat = st.number_input("Dropoff Lat", value=40.75)
                    dlon = st.number_input("Dropoff Lon", value=-73.98)
                
                if st.form_submit_button("Run Prediction"):
                    dist = haversine(plat, plon, dlat, dlon)
                    # Simple dummy input for time/passengers
                    query = np.array([[plon, plat, dlon, dlat, 1, 12, 0, dist]])
                    res = st.session_state.model.predict(query)
                    
                    if st.session_state.problem_type == "Regression":
                        st.header(f"Estimated Fare: ${res[0]:.2f}")
                    else:
                        st.header(f"Fare Class: {'High' if res[0] == 1 else 'Low'}")
        else:
            st.error("Model not available.")

else:
    st.info("Please upload the Uber CSV file in the sidebar to activate the pipeline.")