import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (r2_score, accuracy_score, mean_squared_error, 
                             mean_absolute_error, classification_report, 
                             confusion_matrix, f1_score, precision_score, recall_score)

# ---------------- UI CONFIG ----------------
st.set_page_config(layout="wide", page_title="Uber ML Pipeline Pro")

st.markdown("""
    <style>
    div.stButton > button {
        width: 100%; border-radius: 4px; height: 3.5em;
        background-color: #f8f9fb; border: 1px solid #d1d5db;
        color: #1f2937; font-weight: 600; white-space: normal; line-height: 1.2;
    }
    div.stButton > button:hover { border-color: #ff4b4b; color: #ff4b4b; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if 'df' not in st.session_state: st.session_state.df = None
if 'cleaned_df' not in st.session_state: st.session_state.cleaned_df = None
if 'model' not in st.session_state: st.session_state.model = None
if 'page' not in st.session_state: st.session_state.page = "Data Input"
if 'features_generated' not in st.session_state: st.session_state.features_generated = False

def haversine(lat1, lon1, lat2, lon2):
    r = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# ---------------- NAVIGATION ----------------
nav_list = ["Data Input", "EDA", "Cleaning", "Features", "Split", "Model Selection", "Training", "K-Fold Validation", "Metrics", "Predict"]
cols = st.columns(len(nav_list))
for i, name in enumerate(nav_list):
    if cols[i].button(name):
        st.session_state.page = name

st.divider()

# ---------------- APP LOGIC ----------------
uploaded_file = st.sidebar.file_uploader("Upload uber.csv", type=["csv"])

if uploaded_file is not None:
    if st.session_state.df is None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.cleaned_df = st.session_state.df.copy()
    
    df = st.session_state.df
    page = st.session_state.page

    # --- Sections 1-8 (Same as previous stable version) ---
    if page == "Data Input":
        st.subheader("📂 Raw Dataset Overview")
        st.dataframe(df.head(10), use_container_width=True)
    
    elif page == "EDA":
        st.subheader("📊 Logical Exploratory Analysis")
        tab1, tab2 = st.tabs(["Distributions", "Relationships"])
        with tab1:
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.histogram(df[df['fare_amount'] < 100], x='fare_amount', title="Fare Distribution"), use_container_width=True)
            c2.plotly_chart(px.box(df, x='passenger_count', y='fare_amount', title="Outliers by Passenger Count"), use_container_width=True)
        with tab2:
            c3, c4 = st.columns(2)
            geo_sample = df.sample(min(10000, len(df)))
            c3.plotly_chart(px.scatter(geo_sample, x='pickup_longitude', y='pickup_latitude', color='fare_amount', title="Spatial Density"), use_container_width=True)
            fig_corr, ax = plt.subplots(); sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', ax=ax); c4.pyplot(fig_corr)

    elif page == "Cleaning":
        st.subheader("🧹 Data Sanitization")
        st.write("**Before Cleaning:**"); st.dataframe(st.session_state.cleaned_df.head(5), use_container_width=True)
        if st.button("Execute Cleaning"):
            c_df = st.session_state.cleaned_df.dropna()
            c_df = c_df[(c_df['fare_amount'] > 0) & (c_df['fare_amount'] < 500)]
            st.session_state.cleaned_df = c_df
            st.success("Cleaned!"); st.write("**After Cleaning:**"); st.dataframe(st.session_state.cleaned_df.head(5), use_container_width=True)

    elif page == "Features":
        st.subheader("📌 Feature Engineering")
        if st.button("Generate Uber Features"):
            c_df = st.session_state.cleaned_df
            c_df['pickup_datetime'] = pd.to_datetime(c_df['pickup_datetime'])
            c_df['hour'] = c_df['pickup_datetime'].dt.hour
            c_df['day_of_week'] = c_df['pickup_datetime'].dt.dayofweek
            c_df['dist_km'] = haversine(c_df['pickup_latitude'], c_df['pickup_longitude'], c_df['dropoff_latitude'], c_df['dropoff_longitude'])
            c_df = c_df[(c_df['dist_km'] > 0) & (c_df['dist_km'] < 100)]
            st.session_state.cleaned_df = c_df
            st.session_state.features_generated = True
            st.success("Features generated!"); st.dataframe(c_df[['fare_amount', 'dist_km', 'hour', 'day_of_week']].head(5), use_container_width=True)

    elif page == "Split":
        st.subheader("✂️ Dataset Splitting")
        tsize = st.slider("Test Size %", 10, 50, 20)
        feats = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'hour', 'day_of_week', 'dist_km']
        if st.session_state.features_generated:
            X, y = st.session_state.cleaned_df[feats], st.session_state.cleaned_df['fare_amount']
            st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(X, y, test_size=tsize/100, random_state=42)
            st.success("Split Complete."); st.write("**Training Set:**"); st.dataframe(st.session_state.X_train.head(5), use_container_width=True)
        else: st.error("Generate features first!")

    elif page == "Model Selection":
        st.subheader("🤖 Algorithm Selection")
        st.session_state.problem_type = st.radio("Task", ["Regression", "Classification"])
        if st.session_state.problem_type == "Regression":
            algo = st.selectbox("Regressor", ["Linear Regression", "Random Forest", "Gradient Boosting", "Ridge", "Lasso", "Decision Tree"])
            models = {"Linear Regression": LinearRegression(), "Random Forest": RandomForestRegressor(n_estimators=50, n_jobs=-1), "Gradient Boosting": GradientBoostingRegressor(), "Ridge": Ridge(), "Lasso": Lasso(), "Decision Tree": DecisionTreeRegressor()}
        else:
            algo = st.selectbox("Classifier", ["Logistic Regression", "Random Forest", "Gradient Boosting", "KNN", "Naive Bayes", "Decision Tree"])
            models = {"Logistic Regression": LogisticRegression(), "Random Forest": RandomForestClassifier(n_estimators=50, n_jobs=-1), "Gradient Boosting": GradientBoostingClassifier(), "KNN": KNeighborsClassifier(), "Naive Bayes": GaussianNB(), "Decision Tree": DecisionTreeClassifier()}
        st.session_state.model = models[algo]

    elif page == "Training":
        st.subheader("🏋️ Model Training")
        train_mode = st.radio("Mode", ["10,000 Sample (Fast)", "Full Split Data"])
        if st.button("🔥 Train Now"):
            with st.spinner("Training..."):
                y_tr = st.session_state.y_train
                if st.session_state.problem_type == "Classification":
                    y_tr = (y_tr > st.session_state.cleaned_df['fare_amount'].median()).astype(int)
                X_f, y_f = (st.session_state.X_train.iloc[np.random.choice(len(st.session_state.X_train), 10000)], y_tr.iloc[np.random.choice(len(y_tr), 10000)]) if "Sample" in train_mode else (st.session_state.X_train, y_tr)
                st.session_state.model.fit(X_f, y_f)
                y_p = st.session_state.model.predict(st.session_state.X_test)
                if st.session_state.problem_type == "Classification":
                    st.metric("Accuracy", f"{accuracy_score((st.session_state.y_test > st.session_state.cleaned_df['fare_amount'].median()).astype(int), y_p)*100:.2f}%")
                else: st.metric("R² Score", f"{r2_score(st.session_state.y_test, y_p):.4f}")
                st.success("Training Complete!")

    elif page == "K-Fold Validation":
        st.subheader("🔄 Cross-Validation")
        k_val = st.number_input("K Value", 2, 10, 5)
        if st.button("Run K-Fold"):
            idx = np.random.choice(len(st.session_state.X_train), 10000)
            y_cv = st.session_state.y_train.iloc[idx]
            if st.session_state.problem_type == "Classification":
                y_cv = (y_cv > st.session_state.cleaned_df['fare_amount'].median()).astype(int)
            scores = cross_val_score(st.session_state.model, st.session_state.X_train.iloc[idx], y_cv, cv=k_val)
            st.metric("Mean CV Score", f"{scores.mean():.4f}")

    # --- 9. METRICS (Restored and Corrected) ---
    elif page == "Metrics":
        st.subheader("📊 Performance Dashboard")
        if st.session_state.model:
            y_pred = st.session_state.model.predict(st.session_state.X_test)
            if st.session_state.problem_type == "Regression":
                m1, m2, m3 = st.columns(3)
                m1.metric("R² Score", f"{r2_score(st.session_state.y_test, y_pred):.4f}")
                m2.metric("MAE", f"${mean_absolute_error(st.session_state.y_test, y_pred):.2f}")
                m3.metric("RMSE", f"${np.sqrt(mean_squared_error(st.session_state.y_test, y_pred)):.2f}")
                c1, c2 = st.columns(2)
                with c1: st.plotly_chart(px.scatter(x=st.session_state.y_test, y=y_pred, labels={'x':'Actual','y':'Predicted'}, title="Regression Fit"), use_container_width=True)
                with c2: st.plotly_chart(px.histogram(x=st.session_state.y_test - y_pred, title="Error Residuals"), use_container_width=True)
            else:
                y_t_bin = (st.session_state.y_test > st.session_state.cleaned_df['fare_amount'].median()).astype(int)
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{accuracy_score(y_t_bin, y_pred)*100:.2f}%")
                m2.metric("Precision", f"{precision_score(y_t_bin, y_pred):.4f}")
                m3.metric("F1 Score", f"{f1_score(y_t_bin, y_pred):.4f}")
                st.plotly_chart(px.imshow(confusion_matrix(y_t_bin, y_pred), text_auto=True, title="Confusion Matrix"), use_container_width=True)
        else: st.warning("Train the model first.")

    # --- 10. PREDICT (FULLY IMPLEMENTED) ---
    elif page == "Predict":
        st.subheader("🔮 Predictive Analytics")
        if st.session_state.model is not None:
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                with col1:
                    p_lon = st.number_input("Pickup Longitude", value=-73.98, format="%.6f")
                    p_lat = st.number_input("Pickup Latitude", value=40.73, format="%.6f")
                    passengers = st.slider("Number of Passengers", 1, 6, 1)
                with col2:
                    d_lon = st.number_input("Dropoff Longitude", value=-73.99, format="%.6f")
                    d_lat = st.number_input("Dropoff Latitude", value=40.75, format="%.6f")
                    hour = st.slider("Hour of Day", 0, 23, 12)
                
                if st.form_submit_button("💰 Estimate Fare"):
                    dist = haversine(p_lat, p_lon, d_lat, d_lon)
                    # Order: lon1, lat1, lon2, lat2, pass, hr, dow(fixed 0), dist
                    query_data = np.array([[p_lon, p_lat, d_lon, d_lat, passengers, hour, 0, dist]])
                    prediction = st.session_state.model.predict(query_data)
                    
                    st.divider()
                    if st.session_state.problem_type == "Regression":
                        st.header(f"Estimated Fare: :green[${prediction[0]:.2f}]")
                    else:
                        st.header(f"Fare Category: :blue[{'High' if prediction[0] == 1 else 'Low'}]")
                    st.info(f"Calculated Trip Distance: {dist:.2f} km")
        else:
            st.error("⚠️ No trained model found. Please go to the 'Training' tab first.")

else: st.info("Upload CSV to start.")