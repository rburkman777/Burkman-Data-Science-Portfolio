# Machine Learning Streamlit App

## Project Overview
This app aims to illustrate some of the basics of machine learning for its users and give them access to some machine learning models.
The user will have the option to use either a decision tree model, a linear regression model, or a k-nearest neighbor model on built-in or self-selected data. The user will then be able to customize hyperparameters to discover how changes in the mechanics of the model can impact model performance. 

## Instructions 

### Link to App: https://machinelearningappappcodepy-jgwqytyvmrs4jkhtqsblpd.streamlit.app/

Alternatively, you can run the app locally by following these instructions: 

### How to Run the App Locally

In order to run the app yourself, open up the MLStreamlitApp folder in visual studio code. You can do that by clicking on the green 'code' button on the repository main page and copy and pasting the URL. After that, go to VSCode and open the Command Palette (accessible via Ctrl + Shift + P (Windows/Linux) or Cmd + Shift + P (Mac)) then type Git: Clone. Press enter. Paste the URL and hit enter. Decide where on your computer you want to save the folder and then open the folder.

Alternatively, you can simply follow these steps: create a new folder in VSCode with the same title as the folder on this page. Then create a folder inside that folder called data. Add the three csv files in the data folder here into the data folder in your own VSCode. Then, copy and paste the code from Machine_Learning_Streamlit_App_Code.py into a seperate file in your folder (make sure the new file is outside your data folder). In this way, you can recreate the folder in VSCode. You just need to have the data files (in a folder called 'data'. These are the data files in the built-in datasets) and the Machine_Learning_Streamlit_App_Code.py files open.

Once you have everything open in VSCode, go to the terminal (you can activate the terminal by pressing the third "button" from the bottom right with circle and a triangle). Once you have the terminal open, type "streamlit run (whatever you named the file).py" into the interface and hit enter (so if you don't change the name, you would type "streamlit run Machine_Learning_Streamlit_App_Code.py"). It should take you to the application. If that does not work, copy the second link that gets produced into the interface and paste it into a different browser than the one the application originally tried to open on.

Also, help yourself to the sample datasets that you can upload into the app if you choose to upload your own file instead of using one of the sample datasets (in the sample_datasets_for_app_uploading folder). 

## App Features 
Three models were used: Linear regression, decision trees, and k-nearest neighbor. See below for how you can tune hyperparameters on both models to explore different outcomes:

Linear Regression: You can explore how scaling impacts model performance. We judge the performance of this model on its r^2 value, coefficients, and several other metrics. 

Decision Trees: You can tune hyperparameters such as depth and minimum sample splits to see how changing these parameters impacts the model. The model will be given an accuracy score, an AUC score, and other performance metrics as well.

KNN: You can explore how distance metrics, k, and scaling the model affect model performance. The model will be given an accuracy score, a classification report, and other performance metrics. 

## References
While creating this project I used some useful resources that I have attatched below. 

Here is a useful reference on decision trees: 
[GrokkingML_Decision Trees (3).pdf](https://github.com/user-attachments/files/26556983/GrokkingML_Decision.Trees.3.pdf)

Here is a useful reference on some of the metrics used to evaluate model performance in the app:
[GrokkingML_Measuring Classification Models-1 (1).pdf](https://github.com/user-attachments/files/26556999/GrokkingML_Measuring.Classification.Models-1.1.pdf)

## Visual Examples
Here are some examples of the kinds of visual you will produce using the app (this shows a visualization of a decision tree you can produce in the app on top and a visualization of a graph plotting k against accuracy on the bottom):

<img width="783" height="394" alt="Screen Shot 2026-04-07 at 8 28 35 PM" src="https://github.com/user-attachments/assets/d27812bd-99f2-448e-8739-d3c9033f0e57" />


<img width="817" height="550" alt="Screen Shot 2026-04-13 at 9 13 18 PM" src="https://github.com/user-attachments/assets/23ead8e8-27d9-4c6c-a76e-bb7772b78350" />

 
