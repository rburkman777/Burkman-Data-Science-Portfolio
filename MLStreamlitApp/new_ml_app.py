import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn import tree
import graphviz

st.title("🤖 Machine Learning Streamlit App")

# -----------------------------
# STEP 1: MODEL SELECTION
# -----------------------------
model_type = st.selectbox(
    "👉 First, choose a model",
    ["Select...", "Linear Regression", "Decision Tree"]
)

if model_type == "Select...":
    st.warning("Please select a model to continue.")
    st.stop()

# -----------------------------
# STEP 2: DATA SOURCE
# -----------------------------
data_option = st.selectbox("Choose data source", ["Upload CSV", "Built-in Dataset"])

df = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if data_option == "Upload CSV":
    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV uploaded successfully!")
    else:
        st.stop()
else:
    st.subheader("📁 Using Built-in Dataset")
    dataset_path = os.path.join(BASE_DIR, "data", "student_admissions_data.csv")

    if not os.path.exists(dataset_path):
        st.error(f"Dataset '{dataset_path}' not found.")
        st.stop()

    df = pd.read_csv(dataset_path)
    st.success("Using built-in dataset: student_admissions_data.csv")

# -----------------------------
# STEP 3: DISPLAY DATA
# -----------------------------
st.write("### 📊 Data Preview")
st.write(df.head())
columns = df.columns.tolist()

# =============================
# 📈 LINEAR REGRESSION
# =============================
if model_type == "Linear Regression":
    st.header("📈 Linear Regression")

    x_column = st.selectbox("Select feature (X)", columns)
    y_column = st.selectbox("Select target (y)", columns)

    if st.button("Train Linear Regression"):
        X = df[[x_column]]
        y = df[y_column]

        model = LinearRegression()
        model.fit(X, y)

        st.success("✅ Model trained!")
        st.write(f"Coefficient: {model.coef_[0]}")
        st.write(f"Intercept: {model.intercept_}")

        df["Predictions"] = model.predict(X)
        st.write("### Predictions")
        st.write(df)

        # 🔮 User input prediction
        st.write("### 🔮 Make a Prediction")
        user_value = st.number_input(f"Enter value for {x_column}")
        if st.button("Predict (Regression)"):
            prediction = model.predict([[user_value]])
            st.success(f"Prediction: {prediction[0]}")

# =============================
# 🌳 DECISION TREE
# =============================
elif model_type == "Decision Tree":
    st.header("🌳 Decision Tree")

    target = st.selectbox("Select Target Column (y)", columns)
    features = st.multiselect("Select Feature Columns (X)", columns)

    if target and features:
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Sidebar settings
        st.sidebar.header("⚙️ Model Settings")
        max_depth = st.slider("Max Depth", 1, 10, 3)
        criterion = st.selectbox("Criterion", ["gini", "entropy"])

        if st.button("Train Decision Tree"):
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                criterion=criterion,
                random_state=42
            )
            model.fit(X_train, y_train)

            st.success("✅ Model trained!")

            # Predictions
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f"### 🎯 Accuracy: {acc:.2f}")

            # Confusion Matrix
            st.write("### 🔢 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # Classification Report
            st.write("### 📄 Classification Report")
            st.text(classification_report(y_test, y_pred))

            # Tree Visualization
            st.write("### 🌳 Decision Tree Visualization")
            dot_data = tree.export_graphviz(
                model,
                feature_names=features,
                class_names=[str(c) for c in y.unique()],
                filled=True
            )
            st.graphviz_chart(graphviz.Source(dot_data))

            # 🔮 User Input for Prediction
            st.write("### 🔮 Make a Prediction")
            user_input = {}
            for feature in features:
                if df[feature].dtype in ["int64", "float64"]:
                    user_input[feature] = st.number_input(feature, value=0.0)
                else:
                    user_input[feature] = st.text_input(feature)
            input_df = pd.DataFrame([user_input])
            if st.button("Predict (Decision Tree)"):
                prediction = model.predict(input_df)
                st.success(f"Prediction: {prediction[0]}")

            # ROC Curve (binary only)
            if len(y.unique()) == 2:
                y_probs = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_probs)
                roc_auc = roc_auc_score(y_test, y_probs)
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                ax2.plot([0, 1], [0, 1], linestyle='--')
                ax2.set_xlabel("False Positive Rate")
                ax2.set_ylabel("True Positive Rate")
                ax2.set_title("ROC Curve")
                ax2.legend()
                st.pyplot(fig2)
            else:
                st.info("ROC curve only works for binary classification.")