
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

st.title("📈 Unsupervised Machine Learning Streamlit App")
st.write("Welcome to my unsupervised machine learning app! This app was created to allow users to explore various machine learning models and the ways that they can be used to analyze data. I hope that you will explore all of the features that this app has to offer. To get started, choose a model type below!")
with st.expander("CLICK HERE to learn more about unsupervised machine learning"):
    st.write("Unsupervised machine learning is a variant of AI that finds patterns and relationships in data without human guidance. Unsupervised machine learning" \
    "differs from supervised machine learning in that unsupervised machine learning does not seek to make predictions about features. Instead, it aims to unveil hidden patterns and relationships from unlabeled, unstructured data.")


st.markdown("-----------------------------------------------------------------")

# -----------------------------
# STEP 1: MODEL SELECTION
# -----------------------------

# we want an interactive feature that allows for selection of different models to explore, so we use a selectbox
model_type = st.selectbox(
    "👉 First, choose a model",
    ["Select...", "Hierarchical Clustering", "PCA (Dimensionality Reduction)", "K-Nearest Neighbors (KNN)"]
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
    st.write("You chose PCC! Now let's get a dataset in order for you. You can upload one or use a built-in dataset. You can set up your data source below.")
    st.write("NOTE: If you want to upload your own dataset, make sure that it meets the following parameters: \n\n * It is a csv file \n\n * The rows above each column of data are labelled \n\n * The data is numeric \n\n See the sample data for an example")
    st.markdown("-----------------------------------------------------------------")
elif model_type == "PCA (Dimensionality Reduction)":
    st.markdown("-----------------------------------------------------------------")
    st.write("You chose PCA (Dimensionality Reduction)! Let's get you a dataset to work with. You can use the built in dataset or upload your own. You can set up your dataset below.")
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
    if model_type == "PCA (Dimensionality Reduction)":
        dataset_path = os.path.join(BASE_DIR, "data", "WineQT.csv") # set up path to sample dataset for decision tree
    elif model_type == "Hierarchical Clustering":
        dataset_path = os.path.join(BASE_DIR, "data", "USArrests.csv") # set up path to sample dataset for decision tree


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

elif model_type == "PCA" and data_option == "Built-in Dataset":
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
# 🌳 HIERARCHICAL CLUSTERING
# =============================

elif model_type == "Hierarchical Clustering":
    st.markdown("-----------------------------------------------------------------")
    st.header("🌳 Hierarchical Clustering")

    st.write(
        "Hierarchical clustering is an unsupervised learning technique that builds a tree-like structure "
        "(called a dendrogram) to group similar data points together. You can visually explore how clusters "
        "form and choose the number of clusters (k)."
    )

    st.markdown("-----------------------------------------------------------------")

    # -------------------------
    # Step 1: Select Features
    # -------------------------
    st.markdown("#### Step One: Select Features")

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    features = st.multiselect(
        "Select feature columns (numeric only)",
        numeric_columns
    )

    if len(features) < 2:
        st.warning("Please select at least 2 features.")
        st.stop()

    X = df[features]

    st.markdown("-----------------------------------------------------------------")

    # -------------------------
    # Step 2: Scale Data
    # -------------------------
    st.markdown("#### Step Two: Scale the Data")

    from sklearn.preprocessing import StandardScaler

    scale_option = st.radio("Scale data before clustering?", ["Yes", "No"])

    if scale_option == "Yes":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    with st.expander("CLICK HERE to learn more about scaling"):
        st.write(
            "Scaling ensures that all features contribute equally to distance calculations. "
            "This is especially important for clustering methods like Ward linkage."
        )

    st.markdown("-----------------------------------------------------------------")

    # -------------------------
    # Step 3: Dendrogram
    # -------------------------
    st.markdown("#### Step Three: View Dendrogram")

    from scipy.cluster.hierarchy import linkage, dendrogram

    Z = linkage(X_scaled, method="ward")

    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Distance")

    st.pyplot(fig)

    with st.expander("CLICK HERE to learn more about the dendrogram"):
        st.write(
            "The dendrogram shows how data points are merged into clusters. "
            "The vertical height represents distance between clusters. "
            "You can use this to help decide the number of clusters (k)."
        )

    st.markdown("-----------------------------------------------------------------")

    # -------------------------
    # Step 4: Choose k
    # -------------------------
    st.markdown("#### Step Four: Select Number of Clusters")

    from sklearn.cluster import AgglomerativeClustering

    k = st.slider("Number of Clusters (k)", 2, 10, 4)

    st.markdown("-----------------------------------------------------------------")

    # -------------------------
    # Run Clustering
    # -------------------------
    if st.button("Run Hierarchical Clustering"):

        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        cluster_labels = model.fit_predict(X_scaled)

        st.success("✅ Clustering completed!")

        # -------------------------
        # Silhouette Score
        # -------------------------
        from sklearn.metrics import silhouette_score

        score = silhouette_score(X_scaled, cluster_labels)

        st.markdown("### 📊 Silhouette Score")
        st.markdown(f"## {score:.4f}")

        with st.expander("CLICK HERE to learn more about silhouette score"):
            st.write(
                "The silhouette score measures how well-separated your clusters are. "
                "Values closer to 1 indicate well-defined clusters, while values near 0 suggest overlap."
            )

        # -------------------------
        # PCA Visualization
        # -------------------------
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        st.markdown("-----------------------------------------------------------------")


        st.markdown("### 📉 PCA Visualization (2D)")

        fig2, ax2 = plt.subplots()
        scatter = ax2.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=cluster_labels,
            cmap='viridis',
            edgecolor='k',
            alpha=0.7
        )

        ax2.set_xlabel("Principal Component 1")
        ax2.set_ylabel("Principal Component 2")
        ax2.set_title("Cluster Visualization (PCA)")

        st.pyplot(fig2)

        # -------------------------
        # Results Table
        # -------------------------
        results = df.copy()
        results["Cluster"] = cluster_labels

        st.markdown("### 📊 Cluster Assignments")
        st.dataframe(results, height=200, use_container_width=True)
        st.markdown("### Cluster Sizes")
        st.write(results["Cluster"].value_counts())

        st.markdown("-----------------------------------------------------------------")



################
# PCA
################


elif model_type == "PCA (Dimensionality Reduction)":
    st.markdown("-----------------------------------------------------------------")
    st.header("📉 Principal Component Analysis (PCA)")

    st.write(
        "PCA is an unsupervised learning technique used to reduce the number of features "
        "while preserving as much variance as possible. It helps visualize high-dimensional data "
        "and understand which features matter most. Follow the steps below and click the 'Run PCA' button at the end to activate the model."
    )

    st.markdown("-----------------------------------------------------------------")
    st.markdown("#### Step One: Select Features")
    st.write("We begin by choosing our features. These are the features of your dataset that the model will perform on. For PCA, you must choose at least two features. ")

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    features = st.multiselect(
        "Select feature columns (numeric only)",
        numeric_columns
    )

    if len(features) < 2:
        st.warning("Please select at least 2 features for PCA.")
        st.stop()

    X = df[features]
    st.markdown("-----------------------------------------------------------------")

    # -------------------------
    # Step 2: Scaling
    # -------------------------
    st.markdown("#### Step Two: Scale the Data")
    st.write("Note: We highly recommend that you scale the data for PCA")
    from sklearn.preprocessing import StandardScaler

    scale_option = st.radio("Scale data before PCA?", ["Yes", "No"])

    if scale_option == "Yes":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    with st.expander("CLICK HERE to learn more about scaling the data"):
        st.write("Scaling the data is the process of transforming the features into similar scales without changing the shape of the data. It is highly recommended that the data be scaled for PCA since it is sensitive to variable scales.")


    st.markdown("-----------------------------------------------------------------")

    # -------------------------
    # Step 3: Choose Components
    # -------------------------
    st.markdown("#### Step Three: Select Number of Components")

    n_components = st.slider("Number of Principal Components", 2, min(10, len(features)), 2)
    with st.expander("CLICK HERE to learn more about the components"):
        st.write("The components are new linear combinations of the data ranked by importance. We can imagine them like artificial axes that rotate and project 'high-dimensional' data (data with a lot of features) into a lower dimensional space. There is a tradeoff between having simplfying the data through dimentionality reduction "
        "and retaining greater information about the data. A higher number of components relative to the number of initial features prioiritzes information retention and accuracy while a lower number prioritizes simplicity.")


    from sklearn.decomposition import PCA
    st.markdown("-----------------------------------------------------------------")

    if st.button("Run PCA"):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        st.success("✅ PCA completed!")

        # -------------------------
        # Explained Variance
        # -------------------------
        st.markdown("### 📊 Explained Variance")
        with st.expander("CLICK HERE to learn more about explained variance"):
            st.write("Explained variance is a measurement of how much variance from the dataset each principal component perserves. A larger pricipial component means that more information was perserved. Each principal component has a certain explained variance, as one can see below. " \
            "We generally want the cumulative variance' (the sums of the principal variants), to be larger (closer to 1) because this means more information was perserved.")


        explained = pca.explained_variance_ratio_
        cumulative = explained.cumsum()

        for i, var in enumerate(explained):
            st.write(f"PC{i+1}: {var:.4f}")

        st.write(f"**Cumulative Variance:** {cumulative[-1]:.4f}")

        st.markdown("-----------------------------------------------------------------")

        # -------------------------
        # Scatter Plot (2D only)
        # -------------------------
        if n_components >= 2:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()

            ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)

            ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
            ax.set_title("PCA Projection")

            st.pyplot(fig)

            with st.expander("CLICK HERE to learn more about this graphic"):
                st.write("The above graph plots our principal components with the higest explained variances. The aim of this graph is to use these components to observe relationships in the data easily that we could not otherwise easily see. What you are looking at is the data " \
                "simplified onto a two-dimensional axis.")

        st.markdown("-----------------------------------------------------------------")

        st.markdown("### 📌 Feature Contributions (Top 2 Principal Components)")
        with st.expander("CLICK HERE to learn more about feature contributions"):
            st.write("Feature contributions are measurements of how each input model impacts the model and principal components. A positive score "
            " means that  A positive loading means that higher values of a given feature push a sample's score up along that component's axis. A negative loading does the opposite. A graph of the impact of each feature on " \
            "each principal component is also present for easier viewing. ")
        st.space(size="small")

        loadings_df = pd.DataFrame(
            pca.components_,
            columns=features,
            index=[f'PC{i+1}' for i in range(n_components)]
        )

        st.dataframe(loadings_df.style.format("{:.3f}"))
        st.space(size="small")

        # Only proceed if at least 2 components exist
        if n_components >= 2:
            fig2, ax2 = plt.subplots(figsize=(8, 5))

        # Select top 2 PCs and transpose for grouped bar chart
        loadings_df.loc[['PC1', 'PC2']].T.plot(
            kind='barh',
            ax=ax2
        )

        ax2.set_title("Feature Contributions: PC1 vs PC2")
        ax2.set_xlabel("Loading Value")
        ax2.set_ylabel("Feature")
        ax2.legend(title="Principal Components")

        st.pyplot(fig2)


        st.markdown("-----------------------------------------------------------------")
        # -------------------------
        # Scree Plot
        # -------------------------
        st.markdown("### 📉 Scree Plot")

        fig3, ax3 = plt.subplots()

        ax3.plot(range(1, len(explained)+1), cumulative, marker='o')
        ax3.set_xlabel("Number of Components")
        ax3.set_ylabel("Cumulative Variance")
        ax3.set_title("Explained Variance")

        st.pyplot(fig3)

        with st.expander("CLICK HERE to learn more about the scree plot"):
            st.write("The scree plot shows you how much explained variance you're gaining with each principal component. If you are not gainging a lot at a certain point, you may want to simplify your model. You might want to look for the 'eblow' in the plot -- a point at which additional components offer limited model improvement.")

        st.markdown("### 📊 Variance Explained by Each Principal Component")

        fig4, ax4 = plt.subplots(figsize=(8, 6))

        components = range(1, len(explained) + 1)

        ax4.bar(
            components,
            explained,
            alpha=0.7,
            color='teal'
        )

        ax4.set_xlabel('Principal Component')
        ax4.set_ylabel('Variance Explained')
        ax4.set_title('Variance Explained by Each Principal Component')

        ax4.set_xticks(components)
        ax4.grid(True, axis='y')

        st.pyplot(fig4)
        with st.expander("CLICK HERE to learn more about this plot"):
            st.write("Theis plot turns our above scree plot into a bar graph and portrays how much each principal component increases model variance.")
