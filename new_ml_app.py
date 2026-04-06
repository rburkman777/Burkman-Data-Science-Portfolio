import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Machine Learning Streamlit App")

st.write("Upload your dataset (CSV):" \
"Note: Please assure that the file contains a label for the independent variable, the dependent variable, and the numeric values below it. Here is an example:")
uploaded_file = st.file_uploader("Upload a CSV")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data")
    st.write(df)

    # Select columns
    columns = df.columns.tolist()
    
    x_column = st.selectbox("Select feature (X)", columns)
    y_column = st.selectbox("Select target (y)", columns)

    if st.button("Train Model"):
        X = df[[x_column]]
        y = df[y_column]

        model = LinearRegression()
        model.fit(X, y)

        st.write("### Model Results")
        st.write(f"Coefficient: {model.coef_[0]}")
        st.write(f"Intercept: {model.intercept_}")

        # Predictions
        predictions = model.predict(X)

        df["Predictions"] = predictions
        st.write("### Predictions")
        st.write(df)



#uploaded_file = st.file_uploader("Upload a CSV")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data")
    st.write(df)

    # Select columns
    columns = df.columns.tolist()



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
st.write("This app alos gives you the chance to experiment with a decision tree.")

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