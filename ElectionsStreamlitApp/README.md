# 🗳️ Elections Streamlit App 

Use the link below to access the app on Streamlit cloud:

### Link to App: [CLICK HERE](https://burkman-data-science-portfolio-zyhpnavjxj7zpptlwmskuy.streamlit.app/)

## ✏️ Project Overview

The app lets you explore congressional election information compiled by FiveThirtyEight in the leadup to the 2022 elections. The data is partially based on real election results.  You can sort the congressional distircts by "partisan voting index" (aka PVI) and by their level of density as categorized by state. From this, you can find information on the voting habits of different states and the trends across congressional districts. The app enables easy visualization of American political data. 

## 📱 App Features 

* Interactive chart that can be adjusted via user input to explore the relationship between distirct characteristics and partisan voting habits
* Interactive information about state-level voting trends that can explored by the user 
* Summary table of the voting patterns of various district types based on population density

## Visual Examples
Here are some examples of the kinds of visual you will produce using the app (picture: a graphic of a chart of electoral districts falling within a certain partisan range produced by the app):

<img width="786" height="498" alt="Screen Shot 2026-05-03 at 1 54 29 AM" src="https://github.com/user-attachments/assets/c1e7f28f-7e2b-4841-a489-72bd569eb62d" />


## How to Load the App via Visual Studio Code

In order to run the app yourself, open up the MLStreamlitApp folder in visual studio code. You can do that by clicking on the green 'code' button on the repository main page and copy and pasting the URL. After that, go to VSCode and open the Command Palette (accessible via Ctrl + Shift + P (Windows/Linux) or Cmd + Shift + P (Mac)) then type Git: Clone. Press enter. Paste the URL and hit enter. Decide where on your computer you want to save the folder and then open the folder.

Once you have everything open in VSCode, go to the terminal (you can activate the terminal by pressing the third "button" from the bottom right with circle and a triangle). Once you have the terminal open, type "streamlit run (whatever you named the file).py" into the interface and hit enter (so if you don't change the name, you would type "streamlit run Elections_Streamlit_App_Code.py"). It should take you to the application. If that does not work, copy the second link that gets produced into the interface and paste it into a different browser than the one the application originally tried to open on.

## References
* The data came from FiveThirtyEight. Here is a link to interesting data from FiveThirtyEight: <br>
[FiveThirtyEight Data](https://data.fivethirtyeight.com/)

* Cook Political Report is a great resource for further research on the nature of congressional distircts: <br>
[Cook Political Report](https://www.cookpolitical.com/pvi-map-and-district-list)

* Here is a guide to Streamlit: <br>
[Streamlit Guide](https://docs.streamlit.io/)




