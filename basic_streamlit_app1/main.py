import streamlit as st
#here we import streamlit and give our application a title
st.title("Congressional Districts in the 2022 Election")

#here we have a descirption of the basic functions of our app as well as of some important information. 
st.write("Welcome to my first Streamlit app! As a political science major,"
"I believe there is a lot to be learned from studying election results. This "
"page was designed to let you interact with this data from FiveThirtyEight and explore"
"trends in American elections. You will be able to filter through data about American congressional districts" \
"in the lead up to the 2022 elections below. You will be able to explore how many "
"different kinds of congressional district are in each state and search for patterns. NOTE ON THE DATA: for the pvi_22 variable, it is " 
"measuring the partisanship of each congressional district. A positive value " 
"means an estimate that the district has a more Democratic partisanship while a negative value indicates" 
"a more Republican congressional district.")

#we will need pandas and seaborn. If these are not installed, use conda to do so. 
import pandas as pd


import seaborn as sns

#here we show off our csv file
st.subheader("Exploring Our Dataset")

# Loard the CSV file
df = pd.read_csv("urbanization-index-2022.csv")


st.write("Here's our data!")
st.dataframe(df)


#this is out first filter: we filter by 'grouping' or the way that FiveThirtyEight characterizes the district based on how urban, suburban, or rural it is. 
#note that we made this a dropdown menu
st.write("Use this menu to look at individual distirct types based on population density")
grouping = st.selectbox("Select a district type", df["grouping"].unique(), index = None)
filtered_df = df[df["grouping"] == grouping]


#Here, we create a slider that lets us filter by partisanship. We first need to establish variables for the 
#minimum and maximum value. Then, we activate the function.
st.write("Use the slider to filter districts by partisanship (`pvi_22`)")
min_pvi = int(df['pvi_22'].min())
max_pvi = int(df['pvi_22'].max())
pvi_range = st.slider(
   "Select PVI range",
   min_value=min_pvi,
   max_value=max_pvi,
   value=(min_pvi, max_pvi)
)


#we apply both filters. In other words, this allows us to filter congressional districts for our variables. 
filtered_df = df[
   (df["grouping"] == grouping) &
   (df["pvi_22"] >= pvi_range[0]) &
   (df["pvi_22"] <= pvi_range[1])
]

st.write(f"Districts in the {grouping} group with PVI between {pvi_range[0]} and {pvi_range[1]}:")
st.dataframe(filtered_df)


import streamlit as st
import matplotlib.pyplot as plt


#here, we create an interactive bar plot that sums up the number of congressional districts from each filtration category in each state. 
#note the use of the "state.counts function."
st.subheader(f"Number of selected districts per state")
state_counts = filtered_df['state'].value_counts().sort_values(ascending=False)


plt.figure(figsize=(10,5))
sns.barplot(x=state_counts.index, y=state_counts.values, palette="viridis")
plt.xticks(rotation=90)
plt.xlabel("State")
plt.ylabel("Number of Districts")
plt.title(f"Number of {grouping} districts per state with PVI {pvi_range[0]} to {pvi_range[1]}")
st.pyplot(plt)

#this chart gives a nice summary of one of the trends you may notice -- urban distrcits tend to be more Democratic
#and are more likely to be located in Democratic states and vice versa. 
st.write("As you may have noticed, more rural and Republican districts are in so-called red states and more "
"Democratic and urban distircts are in so-called blue states. It turns out that rural areas tend to lean Republican "
"and urban areas tend to lean Democratic. On the other hand, you may also find some exceptions to this. Try looking "
"at suburban, Republican districts -- you will notice there are actually a lot of these that exist Florida and Texas!")


st.bar_chart(df, x = 'pvi_22', y = 'grouping', x_label = "Partisanship", y_label = "District Category")
