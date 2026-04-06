import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

# Stop execution until user selects a model
if model_type == "Select...":
    st.warning("Please select a model to continue.")
    st.stop()

# -----------------------------
# STEP 2: FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### 📊 Data Preview")
    st.write(df)

    columns = df.columns.tolist()

    # =============================
    # 📈 LINEAR REGRESSION SECTION
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

            # Model results
            st.write(f"Coefficient: {model.coef_[0]}")
            st.write(f"Intercept: {model.intercept_}")

            # Predictions on dataset
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
    # 🌳 DECISION TREE SECTION
    # =============================
    elif model_type == "Decision Tree":
        st.header("🌳 Decision Tree")

        target = st.selectbox("Select Target Column (y)", columns)
        features = st.multiselect("Select Feature Columns (X)", columns)

        if target and features:

            X = df[features]
            y = df[target]

            # Train/Test Split
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

                # Accuracy
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

                    from sklearn.metrics import roc_curve, roc_auc_score

# Get the predicted probabilities for the positive class (malignant)
y_probs = model.predict_proba(X_test)[:, 1]

# Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Compute the Area Under the Curve (AUC) score
roc_auc = roc_auc_score(y_test, y_probs)
print(f"ROC AUC Score: {roc_auc:.2f}")

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Guess') # Plotting 50% line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()