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



uploaded_file = st.file_uploader("Upload a CSV")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data")
    st.write(df)

    # Select columns
    columns = df.columns.tolist()






from sklearn.model_selection import train_test_split

# Split dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Display classification report
print(classification_report(y_test, y_pred))

# Import graphviz and export the decision tree to dot format for visualization
import graphviz
from sklearn import tree  # Ensure to import the tree module from sklearn

dot_data = tree.export_graphviz(model, feature_names=X_train.columns,
                                class_names=["Not_Survived", "Survived"],
                                filled=True)

# Generate and display the decision tree graph
graph = graphviz.Source(dot_data)
graph

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 3, 4],
    'class_weight' : [None, 'balanced']
}

# Initialize the Decision Tree classifier
dtree = DecisionTreeClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator = dtree, # Reference model to use
                           param_grid = param_grid, # Import param_grid
                           cv = 5, # Number of folds to cross validate
                           scoring='f1', # Metric to tune for
                           verbose = 3 # Change the value of verbose to see more/less of model process
                           )

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Get the best estimator
best_dtree = grid_search.best_estimator_

# Predict on the test set
y_pred = best_dtree.predict(X_test)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import roc_curve, roc_auc_score

# Get the predicted probabilities for the positive class (survived)
y_probs = model.predict_proba(X_test)[:, 1]

y_probs_tuned = best_dtree.predict_proba(X_test)[:, 1]

# Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
fpr_tuned, tpr_tuned, thresholds_tuned = roc_curve(y_test, y_probs_tuned)


# Compute the Area Under the Curve (AUC) score
roc_auc = roc_auc_score(y_test, y_probs)
roc_auc_tuned = roc_auc_score(y_test, y_probs_tuned)

print(f"DT ROC AUC Score: {roc_auc:.2f}")
print(f"DT-Tuned ROC AUC Score: {roc_auc_tuned:.2f}")


# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, lw=2, label=f'DT ROC Curve (AUC = {roc_auc:.2f})')
plt.plot(fpr_tuned, tpr_tuned, lw=2, label=f'DT-Tuned ROC Curve (AUC = {roc_auc_tuned:.2f})')
plt.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Guess') # Plotting 50% line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()




