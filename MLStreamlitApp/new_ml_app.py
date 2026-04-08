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
st.write("Welcome to my machine learning app! This app was created to allow users to explore various machine learning models and the ways that they can be used to learn about data. Let's get started!")
st.markdown("-----------------------------------------------------------------")
# -----------------------------
# STEP 1: MODEL SELECTION
# -----------------------------

model_type = st.selectbox(
    "👉 First, choose a model",
    ["Select...", "Linear Regression", "Decision Tree"]
)


with st.expander("CLICK HERE to learn more about each model type"):
    st.write("Linear Regressions: There are models that can tell you about the relationship between two variables. Specifically, we learn whether an increase in one variable leads to an increase or decrease in the other variable. \n\n Decision Trees: Decision trees are a kind of machine learning that make a series of decisions using yes or no questions.")


if model_type == "Select...":
    st.warning("Please select a model to continue.")
    st.stop()

# -----------------------------
# STEP 2: DATA SOURCE
# -----------------------------
if model_type == "Linear Regression":
    st.markdown("-----------------------------------------------------------------")
    st.write("You chose linear regression! Now let's get a dataset in order for you. You can upload one or use our built-in dataset or upload your own.")
    st.write("NOTE: If you want to upload your own dataset, make sure that it meets the following parameters: \n\n *It is a csv file \n\n *The rows above each column of data are labelled \n\n *The data is numeric")
    st.markdown("-----------------------------------------------------------------")
else:
    st.markdown("-----------------------------------------------------------------")
    st.write("You chose decision tree! Let's get you a dataset to work with. You can use the built in dataset or upload your own")
    st.write("NOTE: If you want to upload your own dataset, make sure that it meets the following parameters: \n\n *It is a csv file \n\n *The rows above each column of data are labelled \n\n *The data is numeric")
    st.markdown("-----------------------------------------------------------------")
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
st.write("Here's a preview of our data")
st.write(df)
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
    st.space(size="small")
    st.markdown("-----------------------------------------------------------------")
    st.space(size="small")

    st.header("🌳 Decision Tree")
    st.space(size="small")

    st.write("You chose to make a decision tree! Great choice! Begin by completing step one of making your decision tree below:")
    st.markdown("-----------------------------------------------------------------")
    st.markdown("#### Step One: Enter your target and predicting features below:")
    st.write("The first step is to choose your features and target. The target is what you want the dataset to make predictions about using the decision tree. Meanwhile, the features are what you want the decision tree to use to make the decisions.")
   
    target = st.selectbox("Select Target Column (y) --> what you want the dataset to make a prediction about", columns)
    features = st.multiselect("Select Feature Columns (X) --> the features you want to use to make the prediction", columns)

if target and features:
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step Two
    st.space(size="medium")
    st.markdown("#### Step Two: Tune your hyperparameters below:")
    st.write("Below is where you may tune your hyperparameters to change the function of the model.")

    with st.expander("CLICK HERE to learn more about the different types of hyperparameters tuning"):
        st.write(
            "Max Depth: Controls how deep the decision tree grows.\n\n"
            "Criterion: Measures split quality. Gini minimizes impurity, while entropy measures information gain."
        )

    # ✅ OUTSIDE expander but INSIDE if-block
    max_depth = st.slider("Max Depth", 1, 10, 3)
    criterion = st.selectbox("Criterion", ["gini", "entropy"])

    # ✅ Train button ALSO inside the if-block
    st.space(size="medium")
    st.markdown("#### Step Three: Press the button below to run the decision tree on your data with the features, targets, and hyperparameters you chose! You will get multiple evaluations of the tree's performance:")
    st.write("NOTE: You can change any of the selections you made above to produce a new decision tree!")
    if st.button("Train Decision Tree"):
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            random_state=42
        )
        model.fit(X_train, y_train)

        st.success("✅ Model trained!")
        st.markdown("## Quick Model Evaluation")
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        if len(y.unique()) == 2:
            y_probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc = roc_auc_score(y_test, y_probs)
        st.write(f"### 🎯 Model Accuracy: {acc:.2f}")
        st.write(f"### 📈 Area Under the Curve (AUC) = {roc_auc:.2f}")
        st.write("To learn more about these metrics and others see the full model evaluation below.")
        st.markdown("-----------------------------------------------------------------")
        st.markdown("## Full Model Evaluation")
        st.write("Below there are three different measurements of the model's performance:" \
        "\n\n 1. Accuracy and Classification Report \n\n 2. Confusion Matrix \n\n 3. ROC Curve \n\n\n\n 4. There is also a " \
        "visualization of the decision tree")
        st.space(size="small")
        st.markdown("-----------------------------------------------------------------")
        st.space(size="small")
        st.markdown("### 1. Accuracy and Classification Report")

        # Predictions
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        st.write(f"### 🎯 Model Accuracy: {acc:.2f}")
        with st.expander("CLICK HERE to learn more about accuracy"):
            st.write(
            "Accuracy is a simple measure of what percentage of times the decision tree correctly predicted the outcome using the features."
        )


        st.space(size="small")

        # Classification Report
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()

        st.write("### 📄 Classification Report")
        st.dataframe(report_df.style.format("{:.2f}"))


        with st.expander("The classification report gives us a number useful metrics. \n\n " \
        "*Precision is essentially the accuracy of your positive predcitions \n\n *Recall is essentially the accuracy of your negative predictions." \
        "\n\n *The F-1 score attempts to measure how well a model performs in both recall and precision. It does this by taking the harmonic mean of precision and recall. " \
        "\n\n *Support is the number of actual occurrences of the class in the specified dataset."):
            st.write()
      
        st.space(size="small")
        st.markdown("-----------------------------------------------------------------")
        st.space(size="small")

        st.markdown("### 2. Confusion Matrix 🔢")


        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        with st.expander("CLICK HERE to learn more the confusion matrix"):
            st.write("The confusion matrix is a table that stores the number of false negatives, false positives, true positives, and true negatives. The tool has its columns that label whether a test is predicted to be positive or negative. " \
            "In the rows it has the true value. So you can compare how the model predicted the variable in the columns with what actually happened in the rows. In each of the four quadrants, we then get our true positives, false positives, true negatives, "
            "and false negatives. The top left box on your screen is true negatives, the bottom right box is true positives, the top right box is false positives, and bottom left box is false negatives. Also explain how this relates to the tree")

        st.space(size="small")
        st.markdown("-----------------------------------------------------------------")
        st.space(size="small")

        st.markdown("### 3. ROC Curve")

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

        with st.expander("CLICK HERE to learn more the ROC curve and AUC"):
            st.write("The ROC curve is a plot of the true positive rate against the false positive rate. The AUC " \
            "measures the ability of the model to distinguish classes, with 1 being perfect and 0.5 being random guessing. " \
            "You will notice that it does this by plotting the true positive rate against the false positive rate. In other the words, the AUC is a measure of the probability that the model will correctly rank a randomly chosen positive example higher than a randomly chosen negative example." \
            "So in short, a higher value here indicates better model performance." \
            "This is a good way to measure overall model performance.")

        st.space(size="small")
        st.markdown("-----------------------------------------------------------------")
        st.space(size="small")

        st.markdown("### 4. Decision Tree Visualization 🌳 ")


        # Tree Visualization
        dot_data = tree.export_graphviz(
            model,
            feature_names=features,
            class_names=[str(c) for c in y.unique()],
            filled=True
        )
        st.graphviz_chart(graphviz.Source(dot_data))

        with st.expander("CLICK HERE to learn more this visualization"):
            st.write("The above visualization is of the decision tree that you produced. Notice the branching nature of it." \
            "You will notice that the tree shows you the information it is using to make decisions. It outputs the gini/entropy of each outcome, the 'samples' refers to the number of samples each node is working with, the 'values' refers" \
            "to the number in each group you have per class in each step, and the 'class' is the target group the model decided to place predictors who follow the path to that node in.")