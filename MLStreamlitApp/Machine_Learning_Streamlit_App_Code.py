import streamlit as st #loading streamlit and all its features
import pandas as pd #loading pandas and all its features
import seaborn as sns # loading seaborn and all its features
import matplotlib.pyplot as plt # loading mathplotlib.pyplot and all its features
import os # importing os 

# below we import a variety of tools that we will use for various project features
from sklearn.linear_model import LinearRegression # needed for linear regression
from sklearn.tree import DecisionTreeClassifier # needed for decision tree
from sklearn.model_selection import train_test_split # needed for multiple models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score # needed to evaluate the models
from sklearn import tree 
import graphviz
from sklearn.neighbors import KNeighborsClassifier # needed for KNN

st.title("🤖 Machine Learning Streamlit App")
st.write("Welcome to my machine learning app! This app was created to allow users to explore various machine learning models and the ways that they can be used to analyze data. I hope that you will explore all of the features that this app has to offer. To get started, choose a model type below!")
st.markdown("-----------------------------------------------------------------")
# -----------------------------
# STEP 1: MODEL SELECTION
# -----------------------------

# we want an interactive feature that allows for selection of different models to explore, so we use a selectbox
model_type = st.selectbox(
    "👉 First, choose a model",
    ["Select...", "Linear Regression", "Decision Tree - Classification", "K-Nearest Neighbors (KNN)"]
)

# we use the expander function to create a drop-down box where users can learn more about different model types
with st.expander("CLICK HERE to learn more about each model type"):
    st.write("Linear Regressions: There are models that can tell you about the relationship between variables. Specifically, we learn whether an increase in one variable leads to an increase or decrease in the the target variable. \n\n Decision Trees: Decision trees are a kind of machine learning that make a series of decisions using yes or no questions. In this model, the decision tree classifies features into binary categories."
    "\n\nK-Nearest Neighbor (KNN): This is a machine learning model that makes predictions about data point classifications based on data point similarities and spatial proximity to neighbors.")

# have an option if a model hasn't been selected using if statements 
if model_type == "Select...":
    st.warning("Please select a model to continue.")
    st.stop()

# -----------------------------
# STEP 2: DATA SOURCE
# -----------------------------

# we use if statements to respond to various user choices the user might make. 
# this code gives the user parameters for various model types
if model_type == "Linear Regression":
    st.markdown("-----------------------------------------------------------------")
    st.write("You chose linear regression 📈! Now let's get a dataset in order for you. You can upload one or use a built-in dataset. You can set up your data source below.")
    st.write("NOTE: If you want to upload your own dataset, make sure that it meets the following parameters: \n\n * It is a csv file \n\n * The rows above each column of data are labelled \n\n * The data is numeric \n\n See the sample data for an example")
    st.markdown("-----------------------------------------------------------------")
elif model_type == "Decision Tree - Classification":
    st.markdown("-----------------------------------------------------------------")
    st.write("You chose decision tree! Let's get you a dataset to work with. You can use the built in dataset or upload your own. You can set up your dataset below.")
    st.write("NOTE: If you want to upload your own dataset, make sure that it meets the following parameters: \n\n * MAKE SURE THAT YOUR DATA HAS A BINARY TARGET. In other words, you need a dataset that has a value you wish to predict that is binary (either 1 or 0) \n\n * It is a csv file \n\n * The rows above each column of data are labelled \n\n * The data is numeric \n\n See the built-in dataset for an example")
    st.markdown("-----------------------------------------------------------------")
elif model_type == "K-Nearest Neighbors (KNN)":
    st.markdown("-----------------------------------------------------------------")
    st.write("You chose K-Nearest Neighbors! Let's get you a dataset to work with. You can use the built in dataset or upload your own")
    st.write("NOTE: If you want to upload your own dataset, make sure that it meets the following parameters: \n\n * Make sure that your dataset has a target feature that consists of classes. A dataset where the classes are binary (meaning they are either 1 or 0) is recommended \n\n * It is a csv file \n\n * The rows above each column of data are labelled \n\n * The data is numeric \n\n See the built-in dataset for an example")
    st.markdown("-----------------------------------------------------------------")
data_option = st.selectbox("Choose data source", ["Upload CSV", "Built-in Dataset"]) # here is another selection box to allow the user to choose whether they upload the data or take the built-in dataset 

df = None # this creates our dataframe variable 
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # this code begins setting up our file pathes for the built in datasets

# we use if statements to direct students to their choice of where to obtain their dataset
if data_option == "Upload CSV":
    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) # creates dataframe for the uploaded data 
        st.success("✅ CSV uploaded successfully!")
    else:
        st.stop()

# we move to an else statement if the user does not choose to upload their own file
else: 
    st.subheader("📁 Using Built-in Dataset") # use a subheader to make the title
    # we again use if and elif statements to filter for the different model types
    if model_type == "Linear Regression":
        dataset_path = os.path.join(BASE_DIR, "data", "medical_insurance_cost.csv") # set up path to sample dataset for linear regression
    elif model_type == "Decision Tree - Classification":
        dataset_path = os.path.join(BASE_DIR, "data", "student_admissions_data.csv") # set up path to sample dataset for decision tree
    elif model_type == "K-Nearest Neighbors (KNN)": 
        dataset_path = os.path.join(BASE_DIR, "data", "diabetes.csv") # set up path to sample dataset for KNN
    
    if not os.path.exists(dataset_path):
        st.error(f"Dataset '{dataset_path}' not found.") # sets up error message if there is an issue
        st.stop()

    df = pd.read_csv(dataset_path) # creates dataframe for the sample, built-in data 
    st.success(f"Using built-in dataset: {os.path.basename(dataset_path)}") # success message for the dataset loading

# -----------------------------
# STEP 3: DISPLAY DATA
# -----------------------------

# we use st.write to display some information

st.write("### 📊 Data Preview")
st.write("Here's a preview of our data")
st.write(df) # we show our data 
columns = df.columns.tolist() # retrieves column names so user can pick from them
# we describe the sample data for the various models here -- give the user details about each feature
if model_type == "Linear Regression" and data_option == "Built-in Dataset":
 with st.expander("CLICK HERE for an explainer on the variables in this dataset"):
            st.write("* charges: this is the medical insurance bill for each patient. It is the target variable of the model. \n\n"
                 "* sex: binary variable where 1 means the patient is a man and 0 means it is a woman \n\n"
                 "* bmi: body mass index \n\n"
                 "* children: how many children the individual has \n\n"
                 "* smoker: binary variable to indicate whether the subject is a smoker or not \n\n"
                 "* southwest: binary variable to indicate whether the subject lives in the southwest region of the country or not \n\n"
                 "* southeast: binary variable to indicate whether the subject lives in the southeast region of the country or not \n\n"
                 "* northwest: binary variable to indicate whether the subject lives in the northwest region of the country or not \n\n"
                "* northeast: binary variable to indicate whether the subject lives in the northeast region of the country or not \n\n"

            )

elif model_type == "Decision Tree - Classification" and data_option == "Built-in Dataset":
    with st.expander("CLICK HERE for an explanation on the variables in this dataset"):
        st.write("* admit: the target variable; whether the subject was admitted or not \n\n"
        "* gre: the subject's GRE score \n\n"
        "* gpa: the subjects grade-point average \n\n"
        "* the subject's class rank")

elif model_type == "Decision Tree - Classification" and data_option == "Built-in Dataset":
    with st.expander("CLICK HERE for an explanation on the variables in this dataset"):
        st.write("* admit: the target variable; whether the subject was admitted or not \n\n"
        "* gre: the subject's GRE score \n\n"
        "* gpa: the subjects grade-point average \n\n"
        "* the subject's class rank")
elif model_type == "K-Nearest Neighbors (KNN)" and data_option == "Built-in Dataset":
    with st.expander("CLICK HERE for an explanation on the variables in this dataset"):
        st.write("* Outcome: the target variable. Whether the patient has diabetes or not \n\n"
                 "* Age: patient age \n\n"
                 "* Pregnancies: number of pregnancies patient has had in their life \n\n"
                 "* Glucose: blood glucose level in patient \n\n"
                 "* BloodPressure: patient blood pressure \n\n"
                 "* SkinThickness: a measure of how thin the patient's skin is \n\n"
                 "* Insulin: patient insulin level \n\n"
                 "* BMI: paitient body mass index \n\n"
                 "* Pedigree: diabetes pedigree function \n\n"
                 "* Age: patient age in years")


# =============================
# 📈 LINEAR REGRESSION
# =============================

# if the model type is linear regression, the model follows this code

if model_type == "Linear Regression":
    st.markdown("-----------------------------------------------------------------")
    st.header("📈 Linear Regression")
    st.write("Linear regressions make predictions about features by writing equations based on how each predicting feature impacts the target feature. Select your target feature and predicting features below as well as decide whether to scale the model. After that, you will get some evaluation metrics on your model's performance.")

    # -------------------------
    # Select target + features
    # -------------------------
    # this is the code for the user to select which features they want as the target feature and which features they want as the predicting features
    y_column = st.selectbox("Select target (y) - this is the feature we are making a prediction about", columns) # we use another select box
    # the mutliselect function lets users employ multiple feautres
    x_columns = st.multiselect(
        "Select feature(s) (X) - these are the feature(s) we are using to make the prediction",
        [col for col in columns if col != y_column] # this removes the feature from the x feature menu that was selected as the y feature
    )

    # this stops the execution if the user does not select at least one feature
    if not x_columns:
        st.warning("Please select at least one feature.")
        st.stop()

    # creates dataframe series for x and y 
    X = df[x_columns]
    y = df[y_column]

    # we import some functions to help us measure the quality of our prediction
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # -------------------------
    # Scaling toggle
    # -------------------------

    # this function helps us set up the ability to scale or unscale the model
    scaling_option = st.radio(
        "Choose data scaling option:",
        ["Unscaled", "Scaled"]
    )
    with st.expander("CLICK HERE to learn more about scaling the data"):
        st.write("Scaling the data is the process of transforming the features into similar scales without changing the shape of the data. This can help to contextualize the coefficients of the features (which measure how much each of the features impact our target variable). It can protect against a variable with a larger magnitude from dominating the model. If you hit the 'scaled' button, the data will be scaled. If you hit the 'unscaled' button, it will not be scaled and left as it is.")

    # -------------------------
    # Train/Test Split FIRST
    # -------------------------
    # let's split our data into testing and training data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling (if selected)
    # next we use if and else statements to set up the scaling feature
    if scaling_option == "Scaled":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw) 
        X_test = scaler.transform(X_test_raw)
    else:
        X_train = X_train_raw
        X_test = X_test_raw
    
    # a button the user can click to train the model
    if st.button("Train Linear Regression Model"):
        with st.expander("CLICK HERE to learn more what we just did here"):
            st.write("This model works by splitting the data into 80% training data to create the regression formula (how the features predict the target feature). It then tests the regression it created on 20% of the data to evaluate performance.")

    # -------------------------
    # Model Training 
    # -------------------------
    # here we set up the regression using the training data
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)

    # -------------------------
    # Predictions
    # -------------------------
    # we use the regression we set up and test it on our testing data (20% of our data)
        y_pred = lin_reg.predict(X_test)

    # -------------------------
    # Metrics
    # -------------------------
    # these prepare useful metrics to evaluate our model
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        st.success(f"✅ Model trained ({scaling_option})")
        # here we write out said metrics for our users to see
        st.markdown("-----------------------------------------------------------------")
        st.write("### 📊 Model Performance")
        st.write(f"R² Score: {r2:.4f}")
        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        with st.expander("CLICK HERE to learn more about these metrics"):
            st.write("* R^2: This is a measure of the overall predictive power of your model. It is how much of the variance in the target variable can be explained the predicting varaibles. The closer to 1 the better. A score of 0 indicates no relationship. Very generally (although this varies depending on what you're measuring) 0.25 indicates a weak relationship and 0.75 indicates a substantial relationship." \
            "\n\n * Mean Squared Error (MSE): measures the average squared difference between estimated values and the actual value, acting as a key indicator of predictive model accuracy. It is calculated by averaging the squared residuals (errors), which penalizes large errors or outliers heavily \n\n" \
            "* Root Mean Squared Error: used to measure the average magnitude of prediction errors in models, calculating the square root of the average squared differences between predicted and observed values. It indicates model performance, with lower values signifying higher accuracy (consider the scale of your data when thinking about that)")

    # -------------------------
    # Coefficients AFTER
    # -------------------------
        # we print out more elements of the model with the following code, incluing the coefficients and the intercept:
        st.markdown("-----------------------------------------------------------------")
        coef_df = pd.DataFrame({
            "Feature": x_columns,
            "Coefficient": lin_reg.coef_
    })
        st.write("### Model Coefficients")
        st.dataframe(coef_df)

        st.write(f"Intercept: {lin_reg.intercept_:.4f}")

        with st.expander("CLICK HERE to learn more about coefficients and the y-intercept"):
            st.write("The coefficient(s) measure how much a unit change in the predictor feature changes the value of the predicted target. NOTE: Explore how changing whether the data is scaled or unscaled affects the coefficients. \n\n The y-intercept is the point on the line/regression at which the predicting feature(s) are equal to zero.")

# -------------------------
# Explanation
# -------------------------
# added this to help users with experimentation
        st.info(
    "Switch between scaled and unscaled data to see how coefficients change.\n\n"
    "Scaling standardizes features (mean = 0, std = 1), which makes coefficients comparable.\n\n"
    "Model performance metrics (R², MSE, RMSE) usually stay similar since we are only scaling the predictors and not the target."
)
        
# =============================
# 🌳 DECISION TREE
# =============================

# let's move to the decision tree. We use an elif statement to specify we are focusing on this model now

# -------------------------
# Intro information
# -------------------------

elif model_type == "Decision Tree - Classification":
    st.space(size="small")
    st.markdown("-----------------------------------------------------------------")
    st.space(size="small")

    st.header("🌳 Decision Tree")
    st.space(size="small")

    # we introduce our user to the section using markdown and write functions
    st.write("You chose to make a decision tree! Great choice! Begin by completing the steps below to set up your model:")
    st.markdown("-----------------------------------------------------------------")
    st.markdown("#### Step One: Enter your target and predicting features below:")
    st.write("The first step is to choose your features and target. The target is what you want the dataset to make predictions about using the decision tree. Meanwhile, the features are what you want the decision tree to use to make the decisions. The tree needs these chosen to work.")

    # we use the selectbox fucntion again to allow users to select predicting features and the target features
    target = st.selectbox("Select Target Column (y) --> what you want the dataset to make a prediction about. It should be a binary feature (a feature of 0s and 1s)", columns)
    features = st.multiselect("Select Feature Columns (X) --> the features you want to use to make the prediction. Do not select the target variable in this box", columns)
 
    # -------------------------
    # Data splitting and processing
    # -------------------------
    
    # we again set up our dataframes and then split our data into training and testing data using the below code
    if target and features:
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # Hyperparameter tuning
    # -------------------------
    
    st.space(size="medium")
    st.markdown("#### Step Two: Tune your hyperparameters below:")
    st.write("Below is where you may tune your hyperparameters to change the function of the model.")

    with st.expander("CLICK HERE to learn more about the different types of hyperparameters tuning"):
        st.write(
            "Max Depth: Controls how deep the decision tree grows, ie the number of splits the decision tree makes to predict classes. Do explore how changing max depth affects our model evaluations \n\n"
            "Criterion: Measures split quality. The Gini index measures the performance of a split by the lack of diversity of outcomes in each group of leaves. We calculate the Gini index based on the probability of picking two outcomes from the same group that are different (so we want a lower value)." 
            " Entropy is also interested in getting the groups of results in the leaves that are similar. However, it measures it in a different way. It compares the purity of leaves based on the probability of drawing a certain combination or sequence of items from the set in each leaf group. \n\n"
            "Min Samples Split: this is the minimum number of samples (the number of data points that each node is working with) required to split an internal node. Smaller values counter underfitting while larger values counter overfitting \n\n"
            "Min Sample Leaf: this is the minimum number of samples required to be present in a leaf node. A higher value counters overfitting \n\n"
            "Class Weight: When one class significantly outnumbers another the tree might otherwise just predict the majority class for everything. Selecting 'balanced' tries to fix this problem by assigning weights inversely proportional to class frequencies"


        )


    max_depth = st.slider("Max Depth", 1, 15, 3) # we use a slider to let the user control the max depth
    criterion = st.selectbox("Criterion", ["gini", "entropy"]) # we use the selectbox function to let them control the criterion
    min_samples_split = st.slider("Min Samples Split", 2, 20, 2) # adds our minimum samples for a split
    min_samples_leaf = st.slider("Min Samples Leaf", 1, 20, 1) # adds our minimum leaves for a sample
    class_weight_option = st.selectbox(
        "Class Weight",
        ["None", "balanced"]
)   # finally, we add our class weight options using a select box function

    # Convert class weight selection
    
    class_weight = None if class_weight_option == "None" else "balanced"
    
    st.space(size="medium")
    st.markdown("#### Step Three: Press the button below to run the decision tree on your data with the features, targets, and hyperparameters you chose! You will get multiple evaluations of the tree's performance:")
    st.write("NOTE: You can change any of the selections you made above to produce a new decision tree!")

    # -------------------------
    # Model setup and evaluation metrics 
    # -------------------------
    
    # we establish our training button and parameters
    if st.button("Train Decision Tree"):
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=42
)
        with st.expander("CLICK HERE to learn more what we are doing here"):
            st.write("This model works by splitting the data into 80% training data to create the model. It then tests the model it created on 20% of the data.")

        model.fit(X_train, y_train) # this fits our testing data based on the model from the training data

        # below we print out the various evaluation metrics used to make the model
        st.success("✅ Model trained!")
        st.markdown("## Quick Model Evaluation")
        y_pred = model.predict(X_test) # this makes predictions based off what the model learned from the testing data
        acc = accuracy_score(y_test, y_pred)
       
        # the roc curve needs the target to be binary, so we confirm that with the len function which measures how many classes there are
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

        # Predictions + accuracy calculation
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) # compares number of correct predictions with number of total predictions to caluclate accuracy
        
        # we output a number of our evaluators below 
        st.write(f"### 🎯 Model Accuracy: {acc:.2f}")
        with st.expander("CLICK HERE to learn more about accuracy"):
            st.write(
            "Accuracy is a simple measure of what percentage of times the decision tree correctly predicted the outcome using the features."
        )

        st.space(size="small")

        # Classification Report
        report_dict = classification_report(y_test, y_pred, output_dict=True) # creates the classification report
        report_df = pd.DataFrame(report_dict).transpose() # this flips the rows and columns to make it more readable

        st.write("### 📄 Classification Report")
        st.dataframe(report_df.style.format("{:.2f}")) # this displays the classification table with correct format

        # we once again use the expander function
        with st.expander("CLICK HERE to learn more about the classification report"):
            st.write(
        "The classification report gives us several useful metrics:\n\n"
        "* Precision: The ratio of the correctly predicted class to the total predicted class. In other words, how well the model minimizes false positives for the class \n\n"
        "* Recall: Recall is the ratio of correctly predicted class to all data in the actual class. In other words, it is how well the model minimuzes false negatives in the class \n\n"
        "* F1-score: A balance between precision and recall that tries to capture how well the model performs on both counts by taking the harmonic mean of recall and precision\n\n"
        "* Support: The number of actual occurrences of each class in the dataset \n\n"
        "* 0 and 1 are the different classes in your model. Note that the precision and recall are for each class respectively \n\n"
        "* Accuracy: The overall accuracy score for the classifier gives a general idea of the model's performance but can be misleading as it considers every correct prediction for all classes (so if a class is very large it can inflate this) \n\n"
        "* Macro Average: This is the average of the metrics for each class. It can be helpful to identify imbalances between classes \n\n"
        "* Weighted Average: This is also an average value for both classes but it also takes it account the support"

    )
      
        st.space(size="small")
        st.markdown("-----------------------------------------------------------------")
        st.space(size="small")

        st.markdown("### 2. Confusion Matrix 🔢")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred) #establishes our matrix
        fig, ax = plt.subplots() 
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax) # sets up the color scheme -- we want darker blue for more responses being in a certain box
        ax.set_xlabel("Predicted") # labels x axis
        ax.set_ylabel("Actual") # labels y axis
        st.pyplot(fig) # sets up our plot for viewer

        with st.expander("CLICK HERE to learn more the confusion matrix"):
            st.write("The confusion matrix is a table that stores the number of false negatives, false positives, true positives, and true negatives. The tool has its columns that label whether a test is predicted to be positive or negative. In the rows it has the true value. So you can compare how the model predicted the variable in the columns with what actually happened in the rows. In each of the four quadrants, we then get our true positives, false positives, true negatives, and false negatives. Where the zeroes interact is the true negatives, where the 1s meet is the true positives, where the 0 is predicted and the 1 is actual is the false negative, and where the 1 is predicted and the 0 is actual is the false positives. We want to maximize true positives and negatives for best model performance.")

        st.space(size="small")
        st.markdown("-----------------------------------------------------------------")
        st.space(size="small")

        st.markdown("### 3. ROC Curve")

        # ROC curve code (for binary only, hence the first line of code here) 
        if len(y.unique()) == 2:
            y_probs = model.predict_proba(X_test)[:, 1] # produces probabilities for each class
            fpr, tpr, _ = roc_curve(y_test, y_probs) # computes points for true positive rate and false positive rate
            roc_auc = roc_auc_score(y_test, y_probs) # computes the AUC score
            fig2, ax2 = plt.subplots() # creates the plot
            ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}") # creates the plot
            ax2.plot([0, 1], [0, 1], linestyle='--') 
            ax2.set_xlabel("False Positive Rate") # labeling x axis
            ax2.set_ylabel("True Positive Rate") # labeling y axis
            ax2.set_title("ROC Curve") # giving it a title
            ax2.legend() # establishing a legend
            st.pyplot(fig2) # plotting
        else:
            st.info("ROC curve only works for binary classification.")

        with st.expander("CLICK HERE to learn more the ROC curve and AUC"):
            st.write("The ROC curve is a plot of the true positive rate against the false positive rate. The AUC " \
            "measures the ability of the model to distinguish classes, with 1 being perfect and 0.5 being random guessing. " \
            "In other the words, the AUC is a measure of the probability that the model will correctly rank a randomly chosen positive example higher than a randomly chosen negative example." \
            "So in short, a higher value here indicates better model performance." \
            "This is a good way to measure overall model performance.  Since the orange line represents random guessing, we want our model (the blue line) to be 'above it' and not to straddle it.")

        st.space(size="small")
        st.markdown("-----------------------------------------------------------------")
        st.space(size="small")

        st.markdown("### 4. Decision Tree Visualization 🌳 ")

        # Tree Visualization
        # we use the code below to help us visualize the decision tree
        dot_data = tree.export_graphviz(
            model,
            feature_names=features,
            class_names=[str(c) for c in y.unique()],
            filled=True
        )
        # outputs our chart
        st.graphviz_chart(graphviz.Source(dot_data))

        with st.expander("CLICK HERE to learn more this visualization"):
            st.write("The above visualization is of the decision tree that you produced. Notice the branching nature of it." \
            " You will notice that the tree shows you the information it is using to make decisions. It outputs the gini/entropy of each outcome, the 'samples' refers to the number of samples each node is working with, the 'values' refers" \
            " to the number in each group you have per class in each step, and the 'class' is the target group the model decided to place predictors who follow the path to that node in.")


# =============================
# 🤝 K-NEAREST NEIGHBORS (KNN)
# =============================

# -------------------------
# Intro information
# -------------------------

# we have another elif function for when the user chooses KNN 
elif model_type == "K-Nearest Neighbors (KNN)":
    st.space(size="small")
    st.markdown("-----------------------------------------------------------------")
    st.space(size="small")

    st.header("🤝 K-Nearest Neighbors (KNN)") # creates our section header
    st.space(size="small") # we use these to make space between parts of the app for looks purposes

    st.write("You chose KNN! This model classifies points based on the majority class of their nearest 'neighbors.' In other" \
    "words, we can imagine the model as sorting different groups together on a graph based on which data points are clustered near each other.")

    st.markdown("-----------------------------------------------------------------")
    st.markdown("#### Step One: Enter your target and predicting features below:")
   
    # we again use the selectbox function to allow the user to pick target and predicting variables
    target = st.selectbox(
        "Select Target Column (y) --> this should be your 'classes' or categorical data. For example, in the sample dataset, the diagnosis status (the 'outcome' variable) is the 'class' the model is measuring",
        columns
    )
    # the multiselect function makes it so we can select more than one item
    features = st.multiselect(
        "Select Feature Columns (X) --> the features you want to use to predict the target variable",
        [col for col in columns if col != target]
    )

    # -------------------------
    # Set up and data splitting 
    # -------------------------
    
    # we establish our dataframes and split our data below (using the same format as for the other models)
    if target and features:
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # -------------------------
        # Hyperparameters, tuning, and scaling 
        # -------------------------
        
        st.space(size="medium")
        st.markdown("#### Step Two: Tune your hyperparameters and prepare data below:")

        k = st.slider("Number of Neighbors (k)", 1, 15, 5)
        with st.expander("CLICK HERE to learn more about k"):
            st.write("k is the number of neighbors that the model considers when making a prediction for a new data point. The algorithm identifies the k training samples closest to the target point and assigns the most frequent class label through majority vote.")

        # we use a selectbox to set up our metric hyperparameter
        st.space(size="small")

        metric = st.selectbox("Distance Metric", ["euclidean", "manhattan"])
        with st.expander("CLICK HERE to learn more about distance metric"):
            st.write("This is how the model calculates distance between points. Euclidean is the most straight forward and simply calculates the distance between points. Manhattan distance calculates the distance in a 'grid-based' format (like calculating the distance of streets as they are positioned around buildings). Euclidiean is the most straight forward but some datasets might make more sense with manhattan.")
        # standscaler can help us to scale this data
        from sklearn.preprocessing import StandardScaler
        st.space(size="small")

        # we use the radio function to give the user the option to scale the data
        scale_option = st.radio(
            "Scale the data? (Recommended for KNN)",
            ["Yes", "No"]
        )
        
        # we again use the expander function to create a dropdown menu to give more information
        with st.expander("CLICK HERE to learn more about scaling the data"):
            st.write("This standardizes the scale of the features so that they all have a mean of 0 and a standard deviation of 1. Pay attention to how having unscaled data in a KNN model can be potentially disruptive.")

        # we use if statements implement data scaling 
        if scale_option == "Yes":
            scaler = StandardScaler() # this is the feature from sklearn that does the rescaling
            X_train = scaler.fit_transform(X_train) # we use it to fix the predictor scales
            X_test = scaler.transform(X_test)

        st.space(size="medium")
        st.markdown("#### Step Three: Train your KNN model")

        # -------------------------
        # Model execution and evaluation
        # -------------------------
        
        # this is the code that activates our model and allows it to classify the groups, including by incorporating our parameters
        if st.button("Train KNN Model"):
            model = KNeighborsClassifier(
                n_neighbors=k,
                metric=metric
            )
            with st.expander("CLICK HERE to learn more about what we are doing here"):
                st.write("We train the model by splitting it up into 'training' data and 'testing' data. We do this "
                "so that we have 80% training data and 20% testing data. We use the training data to help the model learn how to make predictions and the testing data to evaluate its performance.")

            model.fit(X_train, y_train) # here is where the model learns from training data
            y_pred = model.predict(X_test) # here is where the model predicts from that learning

            st.success("✅ Model trained!")

            st.markdown("## Quick Model Evaluation")

            acc = accuracy_score(y_test, y_pred) # calculates our accuracy score
            st.write(f"### 🎯 Model Accuracy: {acc:.2f}") # displays our accuracy score
            with st.expander("CLICK HERE to learn more this accuracy"):
                st.write("Model accuracy measures the proportion of features that the model correctly classifies.")

            st.markdown("-----------------------------------------------------------------")
            st.markdown("## Full Model Evaluation")
            st.write("Below there are  different measurements of the model's performance:" \
            "\n\n 1. Classification Report \n\n 2. Confusion Matrix \n\n 3. Accuracy vs k Graph")
            st.space(size="small")
            st.markdown("-----------------------------------------------------------------")
            # Classification Report
            report_dict = classification_report(y_test, y_pred, output_dict=True) # prepares our classification report
            report_df = pd.DataFrame(report_dict).transpose() # fixes the format of the classification table

            st.write("### 📄 1. Classification Report")
            st.dataframe(report_df.style.format("{:.2f}")) # this displays the classification table with correct format
            with st.expander("CLICK HERE to learn more about the classification report"):
                st.write(
        "The classification report gives us several useful metrics:\n\n"
        "* Precision: The ratio of the correctly predicted class to the total predicted class. In other words, how well the model minimizes false positives for the class \n\n"
        "* Recall: Recall is the ratio of correctly predicted class to all data in the actual class. In other words, it is how well the model minimuzes false negatives in the class \n\n"
        "* F1-score: A balance between precision and recall that tries to capture how well the model performs on both counts by taking the harmonic mean of recall and precision\n\n"
        "* Support: The number of actual occurrences of each class in the dataset \n\n"
        "* 0 and 1 are the different classes in your model. Note that the precision and recall are for each class respectively \n\n"
        "* Accuracy: The overall accuracy score for the classifier gives a general idea of the model's performance but can be misleading as it considers every correct prediction for all classes. That can inflate this value is there a huge amount of one class  \n\n"
        "* Macro Average: This is the average of the values for each class. It can be helpful in identifying imbalances between classes \n\n"
        "* Weighted Average: This is also an average value for both classes but it also takes it account the support"

    )
            st.markdown("-----------------------------------------------------------------")

            # Confusion Matrix
            st.markdown("### 🔢 2. Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred) # prepares our confusion matrix
            fig, ax = plt.subplots() 
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax) # sets up color scheme and annotations
            ax.set_xlabel("Predicted") # x axis label
            ax.set_ylabel("Actual") # y axis label
            st.pyplot(fig) # plots our matrix
            with st.expander("CLICK HERE to learn more the confusion matrix"):
                st.write("The confusion matrix is a table that stores the number of false negatives, false positives, true positives, and true negatives. The tool has its columns that label whether a test is predicted to be positive or negative. In the rows it has the true value. So you can compare how the model predicted the variable in the columns with what actually happened in the rows. In each of the four quadrants, we then get our true positives, false positives, true negatives, and false negatives. Where the zeroes interact is the true negatives, where the 1s meet is the true positives, where the 0 is predicted and the 1 is actual is the false negative, and where the 1 is predicted and the 0 is actual is the false positives. We want to maximize true positives and negatives for best model performance."
                "NOTE: If you have multiple classes, the same rules as above apply. True positives will still be where the prediction of the class and the actual class intercept, etc.")


            # -------------------------
            # Accuracy vs K Graph
            # -------------------------
            st.markdown("-----------------------------------------------------------------")
            st.markdown("### 📊 3. Accuracy vs. Number of Neighbors (k)")

            # Define a range of k values to explore (odd numbers only)
            k_values = range(1, 20, 2)
            accuracies = []

            # Loop through different values of k (through the model)
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

            with st.expander("CLICK HERE to learn more this graph"):
                st.write("You might have noticed from experimenting above that accuracy and k are often related. The vertical line shows oyur current k value. Feel free to explore how " \
                "adjusting some of the other model parameters impacts this relationship. You may notice that changing the scale of the data has an impact on this relationship -- go explore!")

