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
from sklearn.neighbors import KNeighborsClassifier

st.title("🤖 Machine Learning Streamlit App")
st.write("Welcome to my machine learning app! This app was created to allow users to explore various machine learning models and the ways that they can be used to learn about data. Let's get started!")
st.markdown("-----------------------------------------------------------------")
# -----------------------------
# STEP 1: MODEL SELECTION
# -----------------------------

model_type = st.selectbox(
    "👉 First, choose a model",
    ["Select...", "Linear Regression", "Decision Tree - Classification", "K-Nearest Neighbors (KNN)"]
)


with st.expander("CLICK HERE to learn more about each model type"):
    st.write("Linear Regressions: There are models that can tell you about the relationship between variables. Specifically, we learn whether an increase in one variable leads to an increase or decrease in the the target variable. \n\n Decision Trees: Decision trees are a kind of machine learning that make a series of decisions using yes or no questions. In this model, the decision tree classifies features into binary categories."
    "\n\nK-Nearest Neighbor (KNN): This is a mahcine learning model that can classify features into groups based on their associated target group. Basically, it tries to make predictions about group classification on features.")


if model_type == "Select...":
    st.warning("Please select a model to continue.")
    st.stop()

# -----------------------------
# STEP 2: DATA SOURCE
# -----------------------------
if model_type == "Linear Regression":
    st.markdown("-----------------------------------------------------------------")
    st.write("You chose linear regression! Now let's get a dataset in order for you. You can upload one or use our built-in dataset or upload your own.")
    st.write("NOTE: If you want to upload your own dataset, make sure that it meets the following parameters: \n\n * It is a csv file \n\n * The rows above each column of data are labelled \n\n * The data is numeric")
    st.markdown("-----------------------------------------------------------------")
elif model_type == "Decision Tree - Classification":
    st.markdown("-----------------------------------------------------------------")
    st.write("You chose decision tree! Let's get you a dataset to work with. You can use the built in dataset or upload your own")
    st.write("NOTE: If you want to upload your own dataset, make sure that it meets the following parameters: \n\n * MAKE SURE THAT YOUR DATA HAS A BINARY TARGET. In other words, you need a dataset that has a value you wish to predict that is binary (either 1 or 0) \n\n * It is a csv file \n\n * The rows above each column of data are labelled \n\n * The data is numeric")
    st.markdown("-----------------------------------------------------------------")
elif model_type == "K-Nearest Neighbors (KNN)":
    st.markdown("-----------------------------------------------------------------")
    st.write("You chose K-Nearest Neighbors! Let's get you a dataset to work with. You can use the built in dataset or upload your own")
    st.write("NOTE: If you want to upload your own dataset, make sure that it meets the following parameters: \n\n * Make sure that your dataset has a target feature that consists of classes. A dataset where the classes are binary (meaning they are either 1 or 0) is recommended \n\n * It is a csv file \n\n * The rows above each column of data are labelled \n\n * The data is numeric")
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

    if model_type == "Linear Regression":
        dataset_path = os.path.join(BASE_DIR, "data", "medical_insurance_cost.csv")
    elif model_type == "Decision Tree - Classification":
        dataset_path = os.path.join(BASE_DIR, "data", "student_admissions_data.csv")
    elif model_type == "K-Nearest Neighbors (KNN)":
        dataset_path = os.path.join(BASE_DIR, "data", "diabetes.csv")
    
    if not os.path.exists(dataset_path):
        st.error(f"Dataset '{dataset_path}' not found.")
        st.stop()

    df = pd.read_csv(dataset_path)
    st.success(f"Using built-in dataset: {os.path.basename(dataset_path)}")
# -----------------------------
# STEP 3: DISPLAY DATA
# -----------------------------
st.write("### 📊 Data Preview")
st.write("Here's a preview of our data")
st.write(df)
columns = df.columns.tolist()
if model_type == "Linear Regression":
 with st.expander("CLICK HERE for an explainer on the variables in this dataset"):
            st.write("* Charges: This is the medical insurance bill for each patient. It is the target variable of the model. \n\n"
                 "* Sex: Binary variable where 1 means the patient is a man and 0 means it is a woman \n\n")

# =============================
# 📈 LINEAR REGRESSION
# =============================
if model_type == "Linear Regression":
    st.header("📈 Linear Regression")

    # -------------------------
    # Select target + features
    # -------------------------
    y_column = st.selectbox("Select target (y)", columns)
    x_columns = st.multiselect(
        "Select feature(s) (X)",
        [col for col in columns if col != y_column]
    )

    if not x_columns:
        st.warning("Please select at least one feature.")
        st.stop()

    X = df[x_columns]
    y = df[y_column]

    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # -------------------------
    # Scaling toggle
    # -------------------------
    scaling_option = st.radio(
        "Choose data scaling option:",
        ["Unscaled", "Scaled"]
    )

    # -------------------------
    # Train model
    # -------------------------
    if st.button("Train Model"):
        
        if scaling_option == "Scaled":
            scaler = StandardScaler()
            X_used = scaler.fit_transform(X)
        else:
            X_used = X

        model = LinearRegression()
        model.fit(X_used, y)

        y_pred = model.predict(X_used)

        # -------------------------
        # Metrics FIRST
        # -------------------------
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)

        st.success(f"✅ Model trained ({scaling_option})")

        st.write("### 📊 Model Performance")
        st.write(f"R² Score: {r2:.4f}")
        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

        # -------------------------
        # Coefficients AFTER
        # -------------------------
        coef_df = pd.DataFrame({
            "Feature": x_columns,
            "Coefficient": model.coef_
        })

        st.write("### Coefficients")
        st.dataframe(coef_df)

        st.write(f"Intercept: {model.intercept_:.4f}")

    # -------------------------
    # Explanation
    # -------------------------
    st.info(
        "Switch between scaled and unscaled data to see how coefficients change.\n\n"
        "Scaling standardizes features (mean = 0, std = 1), which makes coefficients comparable.\n\n"
        "Model performance metrics (R², MSE, RMSE) usually stay similar."
    )
# =============================
# 🌳 DECISION TREE
# =============================
elif model_type == "Decision Tree - Classification":
    st.space(size="small")
    st.markdown("-----------------------------------------------------------------")
    st.space(size="small")

    st.header("🌳 Decision Tree")
    st.space(size="small")

    st.write("You chose to make a decision tree! Great choice! Begin by completing step one of making your decision tree below:")
    st.markdown("-----------------------------------------------------------------")
    st.markdown("#### Step One: Enter your target and predicting features below:")
    st.write("The first step is to choose your features and target. The target is what you want the dataset to make predictions about using the decision tree. Meanwhile, the features are what you want the decision tree to use to make the decisions.")
   
    target = st.selectbox("Select Target Column (y) --> what you want the dataset to make a prediction about. It should be a binary feature (a feature of 0s and 1s)", columns)
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
            "Max Depth: Controls how deep the decision tree grows, ie the number of splits the decision tree makes to predict classes \n\n"
            "Criterion: Measures split quality. The Gini index measures the performance of a split by the lack of diversity of outcomes in each group of leaves. We calculate the Gini index based on the probability of picking two outcomes from the same group that are different (so we want a lower value)." 
            " Entropy is also interested in getting the groups of results in the leaves that are similar. However, it measures it in a different way. It compares the purity of leaves based on the probability of drawing a certain combination or sequence of items from the set in each leaf group." 


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


        with st.expander("CLICK HERE to learn more about the classification report"):
            st.write(
        "The classification report gives us several useful metrics:\n\n"
        "* Precision: The ratio of the correctly predicted class to the total predicted class. In other words, how well the model minimizes false positives for the class \n\n"
        "* Recall: Recall is the ratio of correctly predicted class to all data in the actual class. In other words, it is how well the model minimuzes false negatives in the class \n\n"
        "* F1-score: A balance between precision and recall that tries to capture how well the model performs on both counts by taking the harmonic mean of recall and precision\n\n"
        "* Support: The number of actual occurrences of each class in the dataset \n\n"
        "* 0 and 1 are the different classes in your model. Note that the precision and recall are for each class respectively \n\n"
        "* Accuracy: The overall accuracy score for the classifier gives a general idea of the model's performance but can be misleading as it considers every correct prediction for both classes \n\n"
        "* Macro Average: This is the average of the values for each class \n\n"
        "* Weighted Average: This is also an average value for both classes but it also takes it account the support"

    )
      
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
            st.write("The confusion matrix is a table that stores the number of false negatives, false positives, true positives, and true negatives. The tool has its columns that label whether a test is predicted to be positive or negative. In the rows it has the true value. So you can compare how the model predicted the variable in the columns with what actually happened in the rows. In each of the four quadrants, we then get our true positives, false positives, true negatives, and false negatives. Where the zeroes interact is the true negatives, where the 1s meet is the true positives, where the 0 is predicted and the 1 is actual is the false negative, and where the 1 is predicted and the 0 is actual is the false positives. We want to maximize true positives and negatives for best model performance.")

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
            "In other the words, the AUC is a measure of the probability that the model will correctly rank a randomly chosen positive example higher than a randomly chosen negative example." \
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


# =============================
# 🤝 K-NEAREST NEIGHBORS (KNN)
# =============================
elif model_type == "K-Nearest Neighbors (KNN)":
    st.space(size="small")
    st.markdown("-----------------------------------------------------------------")
    st.space(size="small")

    st.header("🤝 K-Nearest Neighbors (KNN)")
    st.space(size="small")

    st.write("You chose KNN! This model classifies points based on the majority class of their nearest neighbors.")

    st.markdown("-----------------------------------------------------------------")
    st.markdown("#### Step One: Enter your target and predicting features below:")

    target = st.selectbox(
        "Select Target Column (y) --> this should be your 'classes' or categorical data. For example, in the sample dataset, the diagnosis status (the 'outcome' variable) is the 'class' the model is measuring",
        columns
    )

    features = st.multiselect(
        "Select Feature Columns (X)",
        [col for col in columns if col != target]
    )

    if target and features:
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        st.space(size="medium")
        st.markdown("#### Step Two: Tune your hyperparameters below:")

        k = st.slider("Number of Neighbors (k)", 1, 15, 5)
        metric = st.selectbox("Distance Metric", ["euclidean", "manhattan"])

        from sklearn.preprocessing import StandardScaler

        scale_option = st.radio(
            "Scale the data? (Recommended for KNN)",
            ["Yes", "No"]
        )

        if scale_option == "Yes":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        st.space(size="medium")
        st.markdown("#### Step Three: Train your KNN model")

        if st.button("Train KNN Model"):
            model = KNeighborsClassifier(
                n_neighbors=k,
                metric=metric
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success("✅ Model trained!")

            st.markdown("## Quick Model Evaluation")

            acc = accuracy_score(y_test, y_pred)
            st.write(f"### 🎯 Model Accuracy: {acc:.2f}")

            if len(y.unique()) == 2:
                y_probs = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_probs)
                st.write(f"### 📈 Area Under the Curve (AUC) = {roc_auc:.2f}")

            st.markdown("-----------------------------------------------------------------")
            st.markdown("## Full Model Evaluation")

            # Classification Report
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()

            st.write("### 📄 Classification Report")
            st.dataframe(report_df.style.format("{:.2f}"))

            # Confusion Matrix
            st.markdown("### 🔢 Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # ROC Curve
            st.markdown("### 📈 ROC Curve")

            if len(y.unique()) == 2:
                fpr, tpr, _ = roc_curve(y_test, y_probs)
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

                # -------------------------
# 📊 Accuracy vs K Graph
# -------------------------
        st.markdown("-----------------------------------------------------------------")
        st.markdown("### 📊 Accuracy vs. Number of Neighbors (k)")

# Define a range of k values to explore (odd numbers only)
        k_values = range(1, 20, 2)
        accuracies = []

# Loop through different values of k
        for k_val in k_values:
            knn_temp = KNeighborsClassifier(n_neighbors=k_val, metric=metric)
            knn_temp.fit(X_train, y_train)
            y_temp_pred = knn_temp.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_temp_pred))

# Plot accuracy vs. number of neighbors (k)
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, accuracies, marker='o')
        plt.title('Accuracy vs. Number of Neighbors (k)')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Accuracy')
        plt.xticks(list(k_values))

# Highlight selected k
        plt.axvline(k, linestyle='--')

# Show in Streamlit
        st.pyplot(plt)