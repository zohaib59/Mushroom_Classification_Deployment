import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="🍄 Mushroom Classifier by Zohaib",
    page_icon="favicon.ico",
    layout="wide"
)

# --- Custom Branding Header ---
st.markdown(
    """
    <style>
        .custom-header {
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 1rem 0;
        }
        .custom-header img {
            height: 65px;
        }
        .custom-header h1 {
            font-size: 2.4rem;
            color: #0e1117;
            margin: 0;
        }
        .stApp {
            background-color: #f5f7fa;
        }
    </style>
    <div class='custom-header'>
        <img src='zohaib_logo.png' alt='Zohaib Logo'>
        <h1>Mushroom Classifier — Built by Zohaib</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# --- File paths ---
data_path = r"C:\Users\zohaib khan\OneDrive\Desktop\USE ME\dump\zk\mushroom.csv"
model_file = "mushroom_model.pkl"
scaler_file = "mushroom_scaler.pkl"
encoder_file = "mushroom_label_encoders.pkl"

# --- Sidebar Navigation ---
st.sidebar.title("🔍 Mushroom Classifier")
st.sidebar.markdown("Crafted with ❤️ by Zohaib")
page = st.sidebar.radio("Select Page:", ["📊 Model Evaluation", "🔮 Predict Mushroom Class"])

# --- Load Data ---
df = pd.read_csv(data_path)
target_col = "class"
X = df.drop(target_col, axis=1)
y = df[target_col]

# --- Encoding ---
categorical_cols = X.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
joblib.dump(label_encoders, encoder_file)

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)
joblib.dump(target_encoder, "mushroom_target_encoder.pkl")

# --- Train/Test Split & Scale ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, scaler_file)

# --- Train or Load Model ---
if os.path.exists(model_file):
    model = joblib.load(model_file)
else:
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, model_file)

# --- Model Evaluation ---
if page == "📊 Model Evaluation":
    st.subheader("📊 Model Evaluation Overview")
    st.markdown("---")
    st.markdown("### 🔍 Preview of Mushroom Dataset")
    st.dataframe(df.head(), use_container_width=True)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    col1, col2 = st.columns(2)
    col1.metric("🧠 Training Accuracy", f"{accuracy_score(y_train, y_pred_train):.3f}")
    col2.metric("🧪 Testing Accuracy", f"{accuracy_score(y_test, y_pred_test):.3f}")

    st.markdown("### 📋 Classification Report")
    st.code(classification_report(y_test, y_pred_test, target_names=target_encoder.classes_), language="text")

    st.markdown("### 🔲 Confusion Matrix")
    fig_cm, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt="d", cmap="YlGnBu", ax=ax,
                xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig_cm)

# --- Live Prediction Interface ---
if page == "🔮 Predict Mushroom Class":
    st.subheader("🔮 Predict Mushroom Class")
    st.markdown("Use the dropdowns below to input the mushroom's characteristics.")

    input_data = {}
    for col in X.columns:
        if col in categorical_cols:
            options = label_encoders[col].classes_.tolist()
            input_data[col] = st.selectbox(f"{col}:", options)
        else:
            input_data[col] = st.number_input(f"{col}:", value=float(df[col].mean()))

    st.markdown("---")
    if st.button("🔮 Predict Now"):
        input_df = pd.DataFrame([input_data])
        label_encoders = joblib.load(encoder_file)
        for col in categorical_cols:
            input_df[col] = label_encoders[col].transform(input_df[col])

        scaler = joblib.load(scaler_file)
        input_scaled = scaler.transform(input_df)

        model = joblib.load(model_file)
        pred = model.predict(input_scaled)[0]
        pred_label = target_encoder.inverse_transform([pred])[0]

        st.subheader("📢 Prediction Result")
        st.success(f"🍄 This mushroom is classified as: **{pred_label}**")
