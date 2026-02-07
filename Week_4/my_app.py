import streamlit as st
# Markdown Hashtag 
st.title("Congressional Districts in the 2022 Election")


st.write("Welcome to my first Streamlit app! As a political science major,"
"I believe there is a lot to be learned from studying election results. This "
"page was designed to let you interact with this data from FiveThirtyEight and explore"
" trends in American elections.")

import pandas as pd

import seaborn as sns

st.subheader("Exploring Our Dataset")

# Loard the CSV file
df = pd.read_csv("urbanization-index-2022.csv")

st.write("Here's our data!")
st.dataframe(df)

st.write("Use this menu to look at individual distirct types based on population density")
grouping = st.selectbox("Select a district type", df["grouping"].unique(), index = None)
filtered_df = df[df["grouping"] == grouping]

st.write(f"People in the {grouping}")
st.dataframe(filtered_df)

st.bar_chart(df, x = 'pvi_22', y = 'grouping')

box_plot1 = sns.boxplot(x = df["urban"], y=df["pvi_22"])
st.pyplot(box_plot1.get_figure())

st.pyplot(box_plot1.get_figure())

import matplotlib