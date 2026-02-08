import streamlit as st
# Markdown Hashtag
st.title("Congressional Districts in the 2022 Election")


st.write("Welcome to my first Streamlit app! As a political science major,"
"I believe there is a lot to be learned from studying election results. This "
"page was designed to let you interact with this data from FiveThirtyEight and explore"
" trends in American elections. NOTE ON THE DATA: for the pvi_22 variable, it is " \
"measuring the partisanship of each congressional district. A positive value " \
"means an estimate that the district has a more Democratic partisanship while a negative value indicates" \
"a more Republican congressional district.")


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


# --- PVI Slider ---
st.write("Use the slider to filter districts by partisanship (`pvi_22`)")
min_pvi = int(df['pvi_22'].min())
max_pvi = int(df['pvi_22'].max())
pvi_range = st.slider(
   "Select PVI range",
   min_value=min_pvi,
   max_value=max_pvi,
   value=(min_pvi, max_pvi)
)


# --- Apply Both Filters ---
filtered_df = df[
   (df["grouping"] == grouping) &
   (df["pvi_22"] >= pvi_range[0]) &
   (df["pvi_22"] <= pvi_range[1])
]


st.write(f"Districts in the {grouping} group with PVI between {pvi_range[0]} and {pvi_range[1]}:")
st.dataframe(filtered_df)


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# --- Bar Plot: Number of districts per state (affected by both filters) ---
st.subheader(f"Number of selected districts per state")
state_counts = filtered_df['state'].value_counts().sort_values(ascending=False)


plt.figure(figsize=(10,5))
sns.barplot(x=state_counts.index, y=state_counts.values, palette="viridis")
plt.xticks(rotation=90)
plt.xlabel("State")
plt.ylabel("Number of Districts")
plt.title(f"Number of {grouping} districts per state with PVI {pvi_range[0]} to {pvi_range[1]}")
st.pyplot(plt)


st.write("As you may have noticed, more rural and Republican districts are in so-called red states and more "
"Democratic and urban distircts are in so-called blue states. It turns out that rural areas tend to lean Republican "
"and urban areas tend to lean Democratic.")


st.bar_chart(df, x = 'pvi_22', y = 'grouping', x_label = "Partisanship", y_label = "District Category")
