import streamlit as st
# Markdown Hashtag 
st.title("Hello, streamlit!")
st.markdown("# Hello, streamlit")


st.write("This is y first Streamlit app.")

if st.button("Click me!"):
    st.write("You clicked the button")
else:
    ("Click the button, see what happens...")


import pandas as pd

st.subheader("Exploring Our Dataset")

# Loard the CSV file
df = pd.read_csv("data/sample_data-1.csv")

st.write("Here's our data!")
st.dataframe(df)

city = st.selectbox("Select a city", df["City"].unique(), index = None)
filtered_df = df[df["City"] == city]

st.write(f"People in the {city}")
st.dataframe(filtered_df)

st.bar_chart(df["Salary"])

import seaborn as sns

box_plot1 = sns.boxplot(x = df["City"], y=df["Salary"])

st.pyplot(box_plot1.get_figure())

import matplotlib