# ⚙️ Unsupervised Machine Learning Streamlit App 

Follow this link to open the app in streamlit cloud. 

### Link to App: [CLICK HERE](https://burkman-data-science-portfolio-iym9h8kjqjgfvfishrzv7i.streamlit.app/)

## ✏️ Project Overview 
This project aims to guide users into the world of unsupervised machine learnings. Users have the opporunity to explore three distinct types of machine learning: hierarchical clustering, principal component analysis, and k-means clustering. It aims to teach users about the different types of unsupervised machine learning while granting users the chance to experiment with the different models. The app is also designed to give users access to these models for their own purposes as well.

## 📱 App Features 
Three models are available in the app: hierarchical clustering, principal component analysis, and k-means clustering. The models and the opporunities associated with them are described below: 

* Hierarchical Clustering: You can explore how scaling the data and changing the number of clusters or the linkage method changes model performance. You can use a dendrogram to help you make those determinations.

* Principal Component Analysis: You can change the number of components to explore how this affects model performance.

* K-Means Clustering: You can also explore a k-means clustering model. You can observe how changing the number of clusters (k) alters model performance.


## Visual Examples
Here is an example of a visual you can create with this graph -- a graphic created with the k-means clustering section:

<img width="402" height="389" alt="Screen Shot 2026-05-02 at 9 25 32 PM" src="https://github.com/user-attachments/assets/2bcb4444-7515-4069-9954-aa07ed25d86c" />



### How to Run the App Locally

In order to run the app yourself, open up the MLUnsupervisedApp folder in visual studio code. You can do that by clicking on the green 'code' button on the repository main page and copy and pasting the URL. After that, go to VSCode and open the Command Palette (accessible via Ctrl + Shift + P (Windows/Linux) or Cmd + Shift + P (Mac)) then type Git: Clone. Press enter. Paste the URL and hit enter. Decide where on your computer you want to save the folder and then open the folder.

Alternatively, you can simply follow these steps: create a new folder in VSCode with the same title as the folder on this page. Then create a folder inside that folder called data. Add the three csv files in the data folder here into the data folder in your own VSCode. Then, copy and paste the code from unsupervised_machine_learning_app_code.py into a seperate file in your folder (make sure the new file is outside your data folder). In this way, you can recreate the folder in VSCode. You just need to have the data files (in a folder called 'data'. These are the data files in the built-in datasets) and the unsupervised_machine_learning_app_code.py files open.

Once you have everything open in VSCode (from either method), go to the terminal (you can activate the terminal by pressing the third "button" from the bottom right with circle and a triangle). Once you have the terminal open, type "streamlit run (whatever you named the file).py" into the interface and hit enter (so if you don't change the name, you would type "streamlit run Machine_Learning_Streamlit_App_Code.py"). It should take you to the application. If that does not work, copy the second link that gets produced into the interface and paste it into a different browser than the one the application originally tried to open on.

Also, help yourself to the sample datasets that you can upload into the app if you choose to upload your own file instead of using one of the sample datasets (in the data_for_user folder). 

### References
While creating this project I used some useful resources that I have attatched below: 

* Here is a useful article on supervised machine learning: <b>
  [IBM Unsupervised Machine Learning Article](https://www.ibm.com/think/topics/unsupervised-learning)
* To get a grasp on some of the practical applications of this app, take a look at this article:
  [biztechmagazine article on the uses of unsupervised machine learning](https://biztechmagazine.com/article/2025/05/what-are-benefits-unsupervised-machine-learning-and-clustering-perfcon)


