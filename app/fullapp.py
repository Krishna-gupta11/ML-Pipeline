import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

# ---------------- UI CONFIG ----------------
st.set_page_config(layout="wide")

st.markdown("""
# 🚀 Professional ML Pipeline Dashboard
""")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")

problem_type = st.sidebar.selectbox(
    "Select Problem Type",
    ["Regression", "Classification"]
)

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ---------------- MAIN ----------------
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # ---------------- AUTO PREPROCESS ----------------
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
            df[col] = df[col].astype("int64") // 10**9
        except:
            pass

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # ---------------- STEPS TABS ----------------
    tabs = st.tabs([
        "📂 Data Input",
        "📊 EDA",
        "🧹 Cleaning",
        "📌 Features",
        "✂️ Split",
        "🤖 Model",
        "🏋️ Training",
        "📈 Metrics",
        "🔮 Predict"
    ])

    # ================= DATA INPUT =================
    with tabs[0]:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        target = st.selectbox("Select Target", df.columns)

        features = st.multiselect(
            "Select Features",
            df.columns,
            default=[c for c in df.columns if c != target]
        )

        df = df[features + [target]]

        # PCA
        st.subheader("PCA View")
        scaled = StandardScaler().fit_transform(df)
        pca = PCA(2).fit_transform(scaled)

        fig = px.scatter(x=pca[:, 0], y=pca[:, 1], title="PCA Projection")
        st.plotly_chart(fig)

    # ================= EDA =================
    with tabs[1]:
        st.subheader("EDA")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Shape:", df.shape)
            st.write(df.describe())

        with col2:
            fig = px.histogram(df, x=target)
            st.plotly_chart(fig)

        # Correlation heatmap
        corr = df.corr()
        fig = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig)

    # ================= CLEANING =================
    with tabs[2]:
        st.subheader("Outlier Removal")

        method = st.selectbox("Method", ["None", "IQR", "Isolation Forest"])

        if method == "IQR":
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1

            mask = ~((df < (Q1 - 1.5 * IQR)) |
                     (df > (Q3 + 1.5 * IQR))).any(axis=1)

            st.write("Outliers:", len(df) - sum(mask))

            if st.button("Remove IQR"):
                df = df[mask]

        elif method == "Isolation Forest":
            iso = IsolationForest(contamination=0.05)
            preds = iso.fit_predict(df)

            mask = preds == 1
            st.write("Outliers:", len(df) - sum(mask))

            if st.button("Remove IF"):
                df = df[mask]

    # ================= FEATURES =================
    with tabs[3]:
        st.subheader("Feature Insights")

        st.write("Columns:", df.columns.tolist())

        fig = px.box(df)
        st.plotly_chart(fig)

    # ================= SPLIT =================
    with tabs[4]:
        st.subheader("Train/Test Split")

        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        st.success("Data split done")

    # ================= MODEL =================
    with tabs[5]:
        st.subheader("Model Selection")

        model_name = st.selectbox(
            "Choose Model",
            ["Linear Regression", "SVM", "Random Forest", "KMeans"]
        )

        if model_name == "Linear Regression":
            model = LinearRegression()

        elif model_name == "SVM":
            kernel = st.selectbox("Kernel", ["linear", "rbf"])
            model = SVR(kernel=kernel)

        elif model_name == "Random Forest":
            n = st.slider("Trees", 50, 200, 100)
            model = RandomForestRegressor(n_estimators=n)

        else:
            k = st.slider("Clusters", 2, 10, 3)
            model = KMeans(n_clusters=k)

    # ================= TRAIN =================
    with tabs[6]:
        st.subheader("Training")

        k = st.slider("KFold", 3, 10, 5)

        if st.button("Train Model"):

            if model_name == "KMeans":
                model.fit(X)
                st.success("KMeans trained")

            else:
                model.fit(X_train, y_train)

                scores = cross_val_score(model, X, y, cv=k)
                st.write("Scores:", scores)

                st.session_state["model"] = model

    # ================= METRICS =================
    with tabs[7]:
        st.subheader("Performance")

        if "model" in st.session_state:

            model = st.session_state["model"]

            if model_name != "KMeans":

                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                st.write("R2 Score:", r2)

                fig = px.scatter(x=y_test, y=y_pred,
                                 labels={"x": "Actual", "y": "Predicted"})
                st.plotly_chart(fig)

    # ================= PREDICTION =================
    with tabs[8]:
        st.subheader("🔮 Make Prediction")

        if "model" in st.session_state:

            model = st.session_state["model"]

            input_data = []

            for col in X.columns:
                val = st.number_input(f"{col}", value=0.0)
                input_data.append(val)

            if st.button("Predict"):
                pred = model.predict([input_data])
                st.success(f"Prediction: {pred[0]}")