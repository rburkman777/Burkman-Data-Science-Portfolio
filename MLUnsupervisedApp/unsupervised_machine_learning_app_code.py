import streamlit as st #loading streamlit and all its features
import pandas as pd #loading pandas and all its features
import seaborn as sns # loading seaborn and all its features
import matplotlib.pyplot as plt # loading mathplotlib.pyplot and all its features
import os # importing os 

st.write("## ⚙️ Unsupervised Machine Learning App ")
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
    ["Select...", "Hierarchical Clustering", "PCA (Dimensionality Reduction)", "K-Means Clustering"]
)


# we use the expander function to create a drop-down box where users can learn more about different model types
with st.expander("CLICK HERE to learn more about each model type"):
    st.write("Hierarchical Clustering: This is a type of machine learning that builds a tree-like hierarchy in order to group similar datapoints together. With hierarchical clustering, you can uncover multi-level structure in unlabelled data and segment data into variable sized groups \n\n"
    "PCA (Principal Component Analysis): This type of machine learning reduces 'high-dimensional data' (aka data with a lot of features) and simplifies by creating new axes based on the data. This method of unsupervised machine learning allows you to break down and examine complex data in a digstable way. \n\n"
    "K-Means Clustering: This type of machine learning uses simplifies complex or multi-dimensional data and then attempts to group the data into clusters. It does this by attempting to find the optimal central point for each cluster. It begins with three random central points and then recalculates the central points until the optimal cluster arrangement is reached.")

# have an option if a model hasn't been selected using if statements 
if model_type == "Select...":
    st.warning("Please select a model to continue.")
    st.stop()

# -----------------------------
# STEP 2: DATA SOURCE
# -----------------------------

# we use if and elif statements to respond to various user choices the user might make. 
# this code gives the user parameters for various model types
if model_type == "Hierarchical Clustering":
    st.markdown("-----------------------------------------------------------------")
    st.write("### You chose Hierarchical Clustering 👑!")
    st.write("Now let's get a dataset in order for you. You can upload one or use a built-in dataset. You can set up your data source below.")
    st.write("NOTE: If you want to upload your own dataset, make sure that it meets the following parameters: \n\n * It is a csv file \n\n * The rows above each column of data are labelled \n\n * The data is numeric \n\n See the sample data for an example")
    st.markdown("-----------------------------------------------------------------")
elif model_type == "PCA (Dimensionality Reduction)":
    st.markdown("-----------------------------------------------------------------")
    st.write("### You chose PCA (Dimensionality Reduction)🔻")
    st.write("Let's get you a dataset to work with. You can use the built in dataset or upload your own. You can set up your dataset below.")
    st.write("NOTE: If you want to upload your own dataset, make sure that it meets the following parameters: \n\n * The data has at least three features \n\n * It is a csv file \n\n * The rows above each column of data are labelled \n\n * The data is numeric \n\n See the built-in dataset for an example")
    st.markdown("-----------------------------------------------------------------")
elif model_type == "K-Means Clustering":
    st.markdown("-----------------------------------------------------------------")
    st.write("### You chose K-Means Clustering ✨!")
    st.write("Let's get you a dataset to work with. You can use the built in dataset or upload your own")
    st.write("NOTE: If you want to upload your own dataset, make sure that it meets the following parameters: \n\n * It is a csv file \n\n * The rows above each column of data are labelled \n\n * The data is numeric \n\n See the built-in dataset for an example")
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
        dataset_path = os.path.join(BASE_DIR, "data", "USArrests.csv")
 # set up path to sample dataset for decision tree
    elif model_type == "K-Means Clustering":
        dataset_path = os.path.join(BASE_DIR, "data", "spotifysongs.csv") # set up path to sample dataset for decision tree

    if not os.path.exists(dataset_path):
        st.error(f"Dataset '{dataset_path}' not found.") # sets up error message if there is an issue
        st.stop()

    df = pd.read_csv(dataset_path) # creates dataframe for the sample, built-in data 
    st.success(f"Using built-in dataset: {os.path.basename(dataset_path)}") # success message for the dataset loading
   
    if model_type == "Hierarchical Clustering" and data_option == "Built-in Dataset":
        df = df.set_index(df.columns[0])
    if df is None:
        st.stop()

# -----------------------------
# STEP 3: DISPLAY DATA
# -----------------------------

# we use st.write to display some information

st.write("### 📊 Data Preview")
if model_type == "Hierarchical Clustering" and data_option == "Built-in Dataset":
    st.write("This is a dataset on crime statistics in the United States by state")
elif model_type == "PCA (Dimensionality Reduction)" and data_option == "Built-in Dataset":
    st.write("This is a dataset on various metrics of wine quality")
elif model_type == "K-Means Clustering" and data_option == "Built-in Dataset":
    st.write("This is a dataset of various Spotify song metrics")

st.write(df) # we show our data 

columns = df.columns.tolist() # retrieves column names so user can pick from them
# we describe the sample data for the various models here -- give the user details about each feature
if model_type == "Hierarchical Clustering" and data_option == "Built-in Dataset":
    with st.expander("CLICK HERE for an explainer on the variables in this dataset"):
            st.write("* State: the state being examined \n\n " 
            "* Murder: the state's murder rate per 100,000 residents \n\n"
            "* Assult: the state's assault rate per 100,000 residents \n\n" 
            "* UrbanPop: the percentage of the state's population living in urban areas \n\n"
            "* Armed Robberies: the state's armed robbery rate per 100,000 residents")

elif model_type == "PCA (Dimensionality Reduction)" and data_option == "Built-in Dataset":
    with st.expander("CLICK HERE for an explanation on the variables in this dataset"):
        st.write("* fixed accidity: the primary natural acids in wine (tartaric, malic, lactic, citric) that do not evaporate \n\n"
        "* volatile acidity: team-distillable acids in wine, primarily acetic acid (vinegar) and ethyl acetate (nail polish remover) \n\n"
        "* citric acid: a weak, organic acid with a sour flavor \n\n"
        "* residual sugar: amount of sugar in the wine \n\n " 
        "* chlorides: an electrolyte and essential mineral \n\n" 
        "* free sulfer dioxide: unbound portion of added or natural sulfur dioxide that protects against oxidation and microbial spoilage. Crucial for wine stability  \n\n"
        "* total sulfur dioxide: acts as a preservative against microbial spoilage and oxidation\n\n"
        "* density: the mass per unit volume of wine\n\n"
        "* pH: the acidity of the wine \n\n"
        "* sulphates: \n\n" 
        "* alcohol: the alcohol by volume of the wine\n\n"
        "* qulaity: a measure of how 'good' the wine is \n\n"
        "* Id: The wine's id number")

elif model_type == "K-Means Clustering" and data_option == "Built-in Dataset":
    with st.expander("CLICK HERE for an explanation on the variables in this dataset"):
        st.write("* artists: the name of the artist(s) of the song \n\n"
        "* track_name: name of the song \n\n"
        "* popularity: how popular the song if on Spotify\n\n"
        "* duration_ms: how long the song is \n\n"
        "* danceability: a numerical metric (0.0 to 1.0) indicating how suitable a track is for dancing based on musical elements like tempo, rhythm stability, beat strength, and regularity \n\n"
        "* energy: a measure from 0.0 to 1.0 representing a track's intensity, speed, and loudness \n\n" 
        "* loudness: how loud the song is \n\n"
        "* speechiness: measures the presence of spoken words in a track on a scale from 0.0 to 1.0 \n\n"
        "* acoutsicness: a confidence metric ranging from 0.0 to 1.0 that indicates whether a track is acoustic"
        "* instrumentalness: measures how instrumental the song is \n\n"
        "* liveness: a rating in Spotify about whether the track was performed live or not\n\n"
        "* valence: a measure from 0.0 to 1.0, developed by Echo Nest and used in Spotify's API, that describes the musical positiveness of a track \n\n" 
        "* tempo: how fast the song moves \n\n"
        "* time_signature: the musical time signature of the song \n\n" )


# =============================
# 🌳 HIERARCHICAL CLUSTERING
# =============================

# we use an elif statement to establish which model type we are using
if model_type == "Hierarchical Clustering":
    st.markdown("-----------------------------------------------------------------")
    st.header("🌳 Hierarchical Clustering")

    st.write(
        "Hierarchical clustering is an unsupervised learning technique that builds a tree-like structure "
        "(called a dendrogram) to group similar data points together. You can visually explore how clusters "
        "form and choose the number of clusters (k). Follow the steps below to set up your model and feel free to change parameters to adjust model results."
    )

    st.markdown("-----------------------------------------------------------------")

    labels = df.index.tolist()
    # -------------------------
    # Step 1: Select Features
    # -------------------------
    # we use st.markdown for titles
    st.markdown("#### Step One: Select Features")

    # this code allows the user to select their desired features from options
    # you will note that it is set to only take numeric inputs
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    # this code creates the multi-select option for users
    features = st.multiselect(
        "Select the features you want the model to use (numeric only)",
        numeric_columns
    )

    # we assuee that the user selects at least two features using the len() function
    if len(features) < 2:
        st.warning("Please select at least 2 features.")
        st.stop()

    # filters data for the features chosen by the user
    X = df[features]

    st.markdown("-----------------------------------------------------------------")

    # -------------------------
    # Step 2: Scale Data
    # -------------------------
    st.markdown("#### Step Two: Scale the Data")
    st.write("Scaling the data is recommended for hierachical clustering")

    from sklearn.preprocessing import StandardScaler

    # st.radio allows us to create a toggle the user can use to choose whether to scale the data
    scale_option = st.radio("Scale data before clustering?", ["Yes", "No"])

    # we use this code to scale the data
    if scale_option == "Yes":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    with st.expander("CLICK HERE to learn more about scaling"):
        st.write(
            "Scaling standardizes the data so that it is all on the same range of size. Scaling ensures that all features contribute equally to distance calculations. "
            "This is especially important for clustering methods like Ward linkage."
        )

    st.markdown("-----------------------------------------------------------------")

    # -------------------------
    # Step 3: Choose Linkage Method
    # -------------------------
    st.markdown("#### Step Three: Choose Linkage Method")
    st.write("This is how the model will determine the distace between your clusters")
    # we set up our selectbox of features
    linkage_method = st.selectbox(
        "Select linkage method",
        ["Select...", "ward", "single", "complete", "average"]
        )

    with st.expander("CLICK HERE to learn more about linkage methods"):
        st.write("These are the ways in which the distance bewteen the clusters can be measured:  \n\n"
            "* Ward: Calculates distance by minimizing the total variance from the cluster mean. It is highly effective at identifying compact, spherical, and similarly sized clusters, though it is more computationally sensitive to noise.\n\n"
            "* Single: Calculates the distance between two clusters as the smallest distance between any single point in one cluster and any single point in the other \n\n"
            "* Complete: Calculates the distance between two clusters as the largest distance between any single point in one cluster and any single point in the other\n\n"
            "* Average: Calculates distance between clusters by computing all pairwise distances between points in clusters (balanced approach that reduces extreme behaviors)"
        )

    st.markdown("-----------------------------------------------------------------")

    # -------------------------
    # Step 4: Dendrogram
    # -------------------------

    # we use the code below to stop the user from moving on until they choose a linkage method

    if linkage_method == "Select...":
        st.warning("Please choose a linkage method to continue.")
        st.stop()
    st.markdown("#### Step Four: View Dendrogram")

    # we use this to import our needed functions
    from scipy.cluster.hierarchy import linkage, dendrogram

    # we use this to plug in our linkeage method and our data to make the dendrogram
    Z = linkage(X_scaled, method=linkage_method)

    # this sets some parameters for our figure
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, ax=ax, labels=labels)
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Distance")

    # this plots our figure

    st.write("This is a visualization of the model based on the inputs you have given thus far. You will notice that it branches out into clusters!")

    st.pyplot(fig)

    with st.expander("CLICK HERE to learn more about the dendrogram"):
        st.write(
            "The dendrogram is a visualization of our tree-strcutured model and shows how data points are split into clusters. You will notice that the various branches of the dendrogram resemble clusters. "
            "The vertical height represents distance between clusters while the horizontal access features indicators of data points. "
            "You can use this to help decide the number of clusters (k)."
        )

    st.markdown("-----------------------------------------------------------------")

    # -------------------------
    # Step 5: Choose k
    # -------------------------
    st.markdown("#### Step Five: Select Number of Clusters (k)")

    # we import another useful function
    from sklearn.cluster import AgglomerativeClustering
    st.write("This controls how many clusters the model will divide your data into. You can use the dendrogram above to estimate the optimal number of clusters.")

    # we use the st.slider function to allow the user to input the number of clusters they want for each run
    k = st.slider("Number of Clusters (k)", 2, 10, 4)

    st.markdown("-----------------------------------------------------------------")

    # -------------------------
    # Run Clustering
    # -------------------------
    # we use a button to activate the model
    if st.button("Run Hierarchical Clustering"):

        # we use if and else statements to allow the model to run for different linkage types
        # we also preselect the metric 
        if linkage_method == "ward":
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage="ward",
                metric="euclidean"
            )
        else:
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage=linkage_method,
                metric="euclidean"
            )

        cluster_labels = model.fit_predict(X_scaled)

        st.success("✅ Clustering completed!")

        # -------------------------
        # Silhouette Score
        # -------------------------
        # we import our silhouette score calculator 
        from sklearn.metrics import silhouette_score

        # this calculates our sihouette score using our data and our clustering data
        score = silhouette_score(X_scaled, cluster_labels)

        st.markdown("### Quick Model Evaluation")
        st.markdown(f"#### 📊 Silhouette Score: {score:.4f}")

        with st.expander("CLICK HERE to learn more about silhouette score"):
            st.write(
                "The silhouette score is a performance metric that measures how well-separated your clusters are. "
                "Values closer to 1 indicate well-defined clusters, while values near 0 suggest overlap. 0.5 is an alright score."
            )
        st.markdown("-----------------------------------------------------------------")

        st.write("#### Full Model Evaluation")
        st.write("1) Principal Components Visualization \n\n" \
        "2) K-Optimization Graph \n\n"
        "3) Model Logistics")
       
       
        # -------------------------
        # PCA Visualization
        # -------------------------
        
        # for this next step, we will need to peform some principal component analysis
        # accordingly, we will import a tool to help us with that 
        from sklearn.decomposition import PCA

        # we can use this to determine the principal components and then to transform the data
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        st.markdown("-----------------------------------------------------------------")

        st.markdown("### 1) 📉 PCA Visualization (2D)")

        # we plot everything onto a visualization of how the hierarchical clustering clusters the data
        # based on user input
        fig2, ax2 = plt.subplots()
        scatter = ax2.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=cluster_labels,
            cmap='viridis',
            edgecolor='k',
            alpha=0.7
        )

        legend = ax2.legend(*scatter.legend_elements(), title="Clusters")
        ax2.add_artist(legend)

        # set axis labels 
        ax2.set_xlabel("Principal Component 1")
        ax2.set_ylabel("Principal Component 2")
        ax2.set_title("Cluster Visualization (PCA)")

        # this actually generates the plot itself in streamlit
        st.pyplot(fig2)

        with st.expander("CLICK HERE to learn more about this graphic"):
            st.write("This plot helps illustrate how the clusters visible on the dendrogram fit together on a two-dimensional plane. It does this by plotting the data points based on the two largest principal components (which are axes identified by the model that simplify data composed of many features)")

        st.markdown("-----------------------------------------------------------------")


        ##################
        #Silhouette Score
        ##################

        st.markdown("### 2) 📈 K Optimization (Silhouette Analysis)")

        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        import numpy as np

        k_range = range(2, 11)
        sil_scores = []

        for k_test in k_range:

            # use SAME linkage the user selected
            if linkage_method == "ward":
                model = AgglomerativeClustering(
                    n_clusters=k_test,
                    linkage="ward",
                    metric="euclidean"
                )
            else:
                model = AgglomerativeClustering(
                    n_clusters=k_test,
                    linkage=linkage_method,
                    metric="euclidean"
                )

            labels = model.fit_predict(X_scaled)

            # silhouette requires more than 1 cluster present
            score = silhouette_score(X_scaled, labels)
            sil_scores.append(score)

        # Plot results
        fig3, ax3 = plt.subplots()
        ax3.plot(list(k_range), sil_scores, marker="o")
        ax3.set_xticks(list(k_range))
        ax3.set_xlabel("Number of Clusters (k)")
        ax3.set_ylabel("Average Silhouette Score")
        ax3.set_title(f"Silhouette Analysis ({linkage_method} linkage)")
        ax3.grid(True, alpha=0.3)

        st.pyplot(fig3)

        # Best k
        best_k = list(k_range)[np.argmax(sil_scores)]
        best_score = max(sil_scores)

        st.success(f"Best k by silhouette: {best_k} (score = {best_score:.3f})")

        with st.expander("CLICK HERE to learn more this graphic"):
            st.write(
            "This graphic runs your model for multiple different possible k clusters under the parameters you chose. This let's you see exactly which number of k clusters"
            " actually optimalizes your silhouette score."
        )

        st.markdown("-----------------------------------------------------------------")

        # -------------------------
        # Results Table
        # -------------------------
        
        # we also created a results table 
        results = df.copy()
        results["Cluster"] = cluster_labels

        st.write("### 3) Model Logistics")
        st.markdown("#### 📊 Cluster Assignments")
        st.write("Here is information about which cluster the model classified each specific point of data into.")
        st.dataframe(results, height=200, use_container_width=True)

        st.markdown("#### Cluster Sizes")
        st.write("Here is information about the size of each cluster in the model.")
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
    st.write("We begin by choosing our features. These are the features of your dataset that the model will perform on. For PCA, you must choose at least three features. ")

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    features = st.multiselect(
        "Select feature columns (numeric only)",
        numeric_columns
    )

    if len(features) < 3:
        st.warning("Please select at least 3 features for PCA.")
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
        st.write(
            "Scaling the data is the process of transforming the features into similar scales without changing the shape of the data. It is highly recommended that the data be scaled for PCA since it is sensitive to variable scales."
        )

    st.markdown("-----------------------------------------------------------------")

    # -------------------------
    # Step 3: Choose Components
    # -------------------------
    st.markdown("#### Step Three: Select Number of Principal Components")

    n_components = st.slider(
        "Number of Principal Components",
        2,
        len(features),
        2
    )

    with st.expander("CLICK HERE to learn more about the principal components"):
        st.write(
            "The principal components are new linear combinations of the data ranked by importance. They project high-dimensional data into lower dimensions."
        )

    from sklearn.decomposition import PCA

    st.markdown("-----------------------------------------------------------------")

    # -------------------------
    # PCA STATE HANDLING (FIX)
    # -------------------------

    run_pca = st.button("Run PCA")

    if run_pca or "X_pca" in st.session_state:

        if run_pca:
            pca = PCA(n_components=n_components)
            st.session_state.pca_model = pca
            st.session_state.X_pca = pca.fit_transform(X_scaled)
            st.session_state.explained = pca.explained_variance_ratio_
            st.session_state.cumulative = st.session_state.explained.cumsum()

        X_pca = st.session_state.X_pca
        explained = st.session_state.explained
        cumulative = st.session_state.cumulative

        st.success("✅ PCA completed!")

        st.markdown("-----------------------------------------------------------------")
        st.write("#### Model Evaluation")
        st.write(
            "1) Visualization of Data \n\n"
            "2) Variance Explained by Each Principal Component \n\n"
            "3) Scree Plot \n\n"
            "4) Feature Contributions"
        )

        st.markdown("-----------------------------------------------------------------")

        # -------------------------
        # Scatter Plot (2D only)
        # -------------------------
        st.markdown("### 1) ✏️ Visualization of Data")

        if n_components >= 2:
            import matplotlib.pyplot as plt
            import numpy as np

            color_options = [
                col for col in df.columns
                if df[col].nunique() <= 12
            ]

            color_feature = st.selectbox(
                "Color PCA plot by feature (≤12 unique values)",
                ["None"] + color_options
            )

            fig, ax = plt.subplots()

            if color_feature == "None":
                ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
            else:
                categories = df[color_feature].astype(str)

                scatter = ax.scatter(
                    X_pca[:, 0],
                    X_pca[:, 1],
                    c=pd.factorize(categories)[0],
                    cmap="tab10",
                    edgecolor="k",
                    alpha=0.8
                )

                legend = ax.legend(
                    *scatter.legend_elements(),
                    title=color_feature
                )
                ax.add_artist(legend)

            ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
            ax.set_title("PCA Projection")

            st.pyplot(fig)

            with st.expander("CLICK HERE to learn more about this graphic"):
                st.write(
                    "This graph shows PCA projection onto the first two components."
                )

        st.markdown("-----------------------------------------------------------------")

        # -------------------------
        # Variance Explained
        # -------------------------
        st.markdown("### 2) 📊 Variance Explained by Each Principal Component")

        for i, var in enumerate(explained):
            st.write(f"#### PC{i+1}: {var:.4f}")

        st.write(f"#### **Cumulative Variance:** {cumulative[-1]:.4f}")

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

        st.markdown("-----------------------------------------------------------------")

        # -------------------------
        # Scree Plot
        # -------------------------
        st.markdown("### 3) 📉 Scree Plot")

        import numpy as np
        from sklearn.decomposition import PCA

        max_components = min(15, len(features))

        if max_components >= 2:
            pca_full = PCA(n_components=max_components)
            X_pca_full = pca_full.fit_transform(X_scaled)

            explained_full = pca_full.explained_variance_ratio_
            cumulative_full = np.cumsum(explained_full)

            fig3, ax3 = plt.subplots()

            ax3.plot(range(1, len(explained_full) + 1), cumulative_full, marker='o')
            ax3.set_xlabel("Number of Components")
            ax3.set_ylabel("Cumulative Variance")
            ax3.set_title("Scree Plot (Up to 15 Components)")
            ax3.grid(True, alpha=0.3)

            st.pyplot(fig3)

        with st.expander("CLICK HERE to learn more about the scree plot"):
            st.write(
                "The scree plot shows how variance accumulates across components."
            )

        st.markdown("-----------------------------------------------------------------")

        # -------------------------
        # Feature Contributions
        # -------------------------
        loadings_df = pd.DataFrame(
            st.session_state.pca_model.components_,
            columns=features,
            index=[f'PC{i+1}' for i in range(n_components)]
        )

        if n_components >= 2:
            fig2, ax2 = plt.subplots(figsize=(8, 5))

            loadings_df.loc[['PC1', 'PC2']].T.plot(
                kind='barh',
                ax=ax2
            )

            ax2.set_title("Feature Contributions: PC1 vs PC2")
            ax2.set_xlabel("Loading Value")
            ax2.set_ylabel("Feature")

            st.markdown("### 4) 📌 Feature Contributions")

            st.pyplot(fig2)

        st.markdown("#### Table of Most Important Features for Each Principal Component")

        st.dataframe(loadings_df.style.format("{:.3f}"))

        with st.expander("CLICK HERE to learn more about feature contributions"):
            st.write(
                "Feature loadings show how variables contribute to principal components."
            )
################
# K-MEANS CLUSTERING
################

# we start by using an elif statement to filter our k-means clustering model
elif model_type == "K-Means Clustering":
    st.markdown("-----------------------------------------------------------------")
    st.header("📍 K-Means Clustering")

    # we use st.write to give a description of what our model does and how to set it up
    st.write(
        "K-Means is an unsupervised learning algorithm that groups data into k clusters "
        "based on similarity. The model assigns each data point to the nearest cluster center. Follow the instructions below to set up the model and then click the button at the end to run it. You can change any of these parameters to experiment with how altering them adjusts the model resutls!"
    )
    
    # st.markdown() is again used to create lines between features
    st.markdown("-----------------------------------------------------------------")

    # -------------------------
    # Step 1: Select Features
    # -------------------------
    st.markdown("#### Step One: Select Features")

    # the code below scans our dataframe, extracts only numeric values, and converts them into a list of column names
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    # we use the st.multiselect() function to allow the user to select multiple features for the model
    features = st.multiselect(
        "Select feature columns (numeric only)",
        numeric_columns
    )
    
    # this lets us force the user to select at least two features to continue (len() counts the number of selected features)
    if len(features) < 2:
        st.warning("Please select at least 2 features.")
        st.stop()

    # this establishes the data we're using as the features the user selected
    X = df[features]

    st.markdown("-----------------------------------------------------------------")

    # -------------------------
    # Step 2: Scale Data
    # -------------------------
    st.markdown("#### Step Two: Scale the Data")
    st.write("We recommend that you scale the data for k-means clustering")
    
    # we scale the data here
    from sklearn.preprocessing import StandardScaler
    
    # we use the st.radio() function to give the user the option to scale the data or not
    scale_option = st.radio("Scale data before clustering?", ["Yes", "No"])

    # we use the st.radio() function to give the user the option to scale the data or not. 
    # The scaler.fit_transform() function can scale our data for us in the event the user chooses to
    if scale_option == "Yes":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X


    with st.expander("CLICK HERE to learn more about scaling"):
        st.write(
            "Scaling standardizes the data so that it is all on the same range of size. Scaling ensures that all features contribute equally to distance calculations. "
            "This is important for K-Means since it relies on distance."
        )

    st.markdown("-----------------------------------------------------------------")

    # -------------------------
    # Step 3: Choose k
    # -------------------------
    st.markdown("#### Step Three: Choose Number of Clusters (k)")

    # we use the st.slider() function to set up our slider
    k = st.slider("Number of Clusters (k)", 2, 10, 3)

    with st.expander("CLICK HERE to learn more about k"):
        st.write("This is the number of clusters that you tell the algorithm to identify with the data. You can play around with different k values to see how changing it affects your model results.")

    st.markdown("-----------------------------------------------------------------")

    # -------------------------
    # Run KMeans
    # -------------------------
    if st.button("Run K-Means Clustering"):

        # we import our model
        from sklearn.cluster import KMeans

        # we input some important information such as the number of clusters (saved as k)
        model = KMeans(n_clusters=k, random_state=42)
       
        # we use this function to run our model on the data
        clusters = model.fit_predict(X_scaled)

        st.success("✅ K-Means clustering completed!")

        # -------------------------
        # Silhouette Score
        # -------------------------
        from sklearn.metrics import silhouette_score

        # we calculate a silhouette score for the data
        score = silhouette_score(X_scaled, clusters)

        # we use st.markdown() to portray out silhouette score inforamtion
        st.markdown("### 📊 Quick Model Evaluation")
        st.markdown(f"#### Silhouette Score: {score:.4f}")

        with st.expander("CLICK HERE to learn more about the silhouette score"):
            st.write("This a measure of model performance telling us how well seperated our clusters are. A score closer to 1 means that the clusters are more well seperated while a score closer to 0 indicates clusters that overlap. A decent score is usually one above 0.5")

        st.markdown("-----------------------------------------------------------------")
        st.markdown("### 📊 Full Model Evaluation")
        st.write("1) Cluster Visualization \n\n" 
        "2) Choosing Optimal k \n\n" \
        "3) Additional Model Info and Logistics")


        # -------------------------
        # PCA Visualization
        # -------------------------
        from sklearn.decomposition import PCA

        # we use this to ensure that there are two principal components in our visualization and run the model
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        st.markdown("-----------------------------------------------------------------")
        st.markdown("### 1) 📉 Cluster Visualization (PCA Projection)")

        # we set up the parameters of our plot below
        fig, ax = plt.subplots()

        # we set up a scatterplot and establish that the clusters will be colored distinctly 
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=clusters,
            cmap='viridis',
            edgecolor='k',
            alpha=0.7
        )

        # we set up axis labels
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("K-Means Clusters (PCA Projection)")

        plt.colorbar(scatter)

        # we use this to set up our plot
        st.pyplot(fig)


        with st.expander("CLICK HERE to learn more about this visualization"):
            st.write("The above graphic plots our data along the axes of the two largest principal components (which are axes generated to simplify high-dimensional data while preserving information about the data) and plots our datapoints. You will notice that the model has assigned the data to clusters based on the number of clusters you assigned in the previous section.")


        st.markdown("-----------------------------------------------------------------")

        # -------------------------
        # Elbow + Silhouette Analysis
        # -------------------------
        st.markdown("### 2) 📉 Choosing Optimal k")

        # these lines set up a number of variables we use to have multiple k values appear in our plot
        ks = range(2, 11)
        wcss = []
        silhouette_scores = []

        # the code below tests different values of k on the model and stores the values of each performance
        for i in ks:
            km = KMeans(n_clusters=i, random_state=42)
            km.fit(X_scaled)
            wcss.append(km.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, km.labels_))

        # we prepare out plots (this makes two side by side plots)
        fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))

        # Code to set up elbow plot
        ax2[0].plot(ks, wcss, marker='o')
        ax2[0].set_title("Elbow Method")
        ax2[0].set_xlabel("k")
        ax2[0].set_ylabel("WCSS")

        # Code to set up silhouette plot
        ax2[1].plot(ks, silhouette_scores, marker='o', color='green')
        ax2[1].set_title("Silhouette Score")
        ax2[1].set_xlabel("k")
        ax2[1].set_ylabel("Score")

        st.pyplot(fig2)

        with st.expander("CLICK HERE to learn more about these graphics and choosing k"):
            st.write(
                "The graph on the left plots the cluster sum of squares against the different values of k. It is a useful tool for 'The Elbow Method.' The Elbow Method helps identify the point where adding more clusters "
                "does not significantly improve model fit (to find this point, we should look at where the graph on the left 'bends.'). \n\n" \
                "The graph on the right shows how different values of k might affect the sihouette score. The silhouette score measures how well-separated the clusters are. Please be encouraged to try out the k value that optimizes your silhouette score using this information!"
            )

        st.markdown("-----------------------------------------------------------------")

        # -------------------------
        # Results Table
        # -------------------------

        # the code below sets up the table for our results
        results = df.copy()
        results["Cluster"] = clusters
            
        st.markdown("### 3) 📊 Additional Model Info & Logistics")
        st.space()
        st.markdown("#### 📊 Cluster Assignments")
        st.write("Here you can see which cluster each of the datapoints got assigned to")
        st.dataframe(results, use_container_width=True)
        st.space()
        st.markdown("#### Cluster Sizes")
        st.write("Here you can see the total number of data points in each cluster")
        st.write(results["Cluster"].value_counts())
