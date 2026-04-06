import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import graphviz

st.title("Decision Tree Machine Learning App")

# -----------------------------
# Upload Dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload your CSV file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### Data Preview")
    st.write(df)

    # -----------------------------
    # Column Selection
    # -----------------------------
    columns = df.columns.tolist()

    target = st.selectbox("Select Target Column (y)", columns)
    features = st.multiselect("Select Feature Columns (X)", columns)

    if target and features:

        X = df[features]
        y = df[target]

        # -----------------------------
        # Train/Test Split
        # -----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # -----------------------------
        # Model Settings
        # -----------------------------
        st.sidebar.header("Model Settings")

        max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
        criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])

        # -----------------------------
        # Train Model
        # -----------------------------
        if st.button("Train Model"):
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                criterion=criterion,
                random_state=42
            )

            model.fit(X_train, y_train)

            st.success("Model trained successfully!")

            # -----------------------------
            # Predictions
            # -----------------------------
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.write(f"### 🎯 Accuracy: {acc:.2f}")

            # -----------------------------
            # Confusion Matrix
            # -----------------------------
            st.write("### 🔢 Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            st.pyplot(fig)

            # -----------------------------
            # Classification Report
            # -----------------------------
            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))

            # -----------------------------
            # Decision Tree Visualization
            # -----------------------------
            st.write("### Decision Tree Visualization")

            dot_data = tree.export_graphviz(
                model,
                feature_names=features,
                class_names=[str(c) for c in y.unique()],
                filled=True
            )

            graph = graphviz.Source(dot_data)
            st.graphviz_chart(graph)

            # -----------------------------
            # User Input for Prediction
            # -----------------------------
            st.write("### Make a Prediction")

            user_input = {}

            for feature in features:
                if df[feature].dtype in ["int64", "float64"]:
                    user_input[feature] = st.number_input(f"{feature}", value=0.0)
                else:
                    user_input[feature] = st.text_input(f"{feature}")

            input_df = pd.DataFrame([user_input])

            if st.button("Predict"):
                prediction = model.predict(input_df)
                st.success(f"Prediction: {prediction[0]}")