import streamlit as st
# Markdown Hashtag 
st.title("Hello, Streamlit!")
st.markdown("# Hello, streamlit")


st.write("This is my first Streamlit app.")

color = st.color_picker("Pick a color", "#00f900")
st.write(f"You picked: {color}")

if st.button("Click me!"):
    st.write("You clicked the button")
else:
    ("Click the button, see what happens...")


import pandas as pd

import seaborn as sns

st.subheader("Exploring Our Dataset")

# Loard the CSV file
df = pd.read_csv("urbanization-index-2022.csv")

st.write("Here's our data!")
st.dataframe(df)

box_plot1 = sns.boxplot(x = df["urban"], y=df["pvi2022"])


city = st.selectbox("Select a city", df["City"].unique(), index = None)
filtered_df = df[df["City"] == city]

st.write(f"People in the {city}")
st.dataframe(filtered_df)

st.bar_chart(df["Urban"])

import seaborn as sns


st.pyplot(box_plot1.get_figure())

import matplotlib