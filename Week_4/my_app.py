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

st.write(f"People in the {grouping}")
st.dataframe(filtered_df)


st.write("Here are some charts highlighting some findings in the data; they both illustrate" \
" that Democrats are more likely to win in urban areas.")











import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("urbanization-index-2022.csv")

st.title("Congressional Districts in the 2022 Election")

# Dropdown to select grouping
grouping = st.selectbox("Select a district type", df["grouping"].unique())
filtered_df = df[df["grouping"] == grouping]

st.write(f"Showing data for: {grouping}")
st.dataframe(filtered_df)

# Slider to select range of pvi_22 values
min_pvi = int(df["pvi_22"].min())
max_pvi = int(df["pvi_22"].max())
pvi_range = st.slider(
    "Select Partisanship Range (PVI)",
    min_value=min_pvi,
    max_value=max_pvi,
    value=(min_pvi, max_pvi)
)

# Filter dataframe based on slider
filtered_df = filtered_df[
    (filtered_df["pvi_22"] >= pvi_range[0]) & (filtered_df["pvi_22"] <= pvi_range[1])
]

st.write(f"Filtered data for PVI between {pvi_range[0]} and {pvi_range[1]}")
st.dataframe(filtered_df)

# Aggregate for bar chart
avg_pvi = filtered_df.groupby("grouping")["pvi_22"].mean()

# Plot bar chart
fig, ax = plt.subplots()
ax.bar(avg_pvi.index, avg_pvi.values)
ax.set_xlabel("District Category")
ax.set_ylabel("Average Partisanship (PVI)")
ax.set_title("Average PVI by District Category (Filtered)")

st.pyplot(fig)














import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Congressional Districts in the 2022 Election")

# Load CSV
df = pd.read_csv("urbanization-index-2022.csv")

st.subheader("Exploring Our Dataset")
st.write("Here's our data!")
st.dataframe(df)

# Dropdown for grouping
grouping = st.selectbox("Select a district type", df["grouping"].unique())
filtered_df = df[df["grouping"] == grouping]

# Slider for PVI range
min_pvi = int(df["pvi_22"].min())
max_pvi = int(df["pvi_22"].max())
pvi_range = st.slider(
    "Select Partisanship Range (PVI)",
    min_value=min_pvi,
    max_value=max_pvi,
    value=(min_pvi, max_pvi)
)

# Filter dataframe by PVI range
filtered_df = filtered_df[
    (filtered_df["pvi_22"] >= pvi_range[0]) & (filtered_df["pvi_22"] <= pvi_range[1])
]

st.write(f"Showing {grouping} districts with PVI between {pvi_range[0]} and {pvi_range[1]}")
st.dataframe(filtered_df)

# Bar chart of filtered data
st.bar_chart(
    data=filtered_df.set_index("grouping")["pvi_22"]
)

# Optional: boxplot
box_plot1 = sns.boxplot(x=filtered_df["urban"], y=filtered_df["pvi_22"])
box_plot1.set_xlabel("District Proportion Urban")
box_plot1.set_ylabel("Democratic Partisanship")
st.pyplot(box_plot1.get_figure())



















st.bar_chart(df, x = 'pvi_22', y = 'grouping', x_label = "Partisanship", y_label = "District Category")

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

sns.boxplot(
    x=df["urban"],
    y=df["pvi_22"],
    ax=ax
)

ax.set_xlabel("District Proportion Urban")
ax.set_ylabel("Democratic Partisanship")

st.pyplot(fig)


import matplotlib


st.subheader("PVI Distribution by District Type")

selected_group = st.selectbox(
    "Select a district type",
    options=df["grouping"].unique()
)


group_df = df[df["grouping"] == selected_group]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))

sns.histplot(
    data=group_df,
    x="pvi_22",
    bins=20,
    ax=ax
)

ax.set_title(f"PVI Distribution for {selected_group} Districts")
ax.set_xlabel("Partisan Voting Index (PVI)")
ax.set_ylabel("Number of Districts")

st.pyplot(fig)


box_plot1 = sns.boxplot(x = df["urban"], y=df["pvi_22"], x_label = "District Proportion Urban", y_label = "Democratic Partisanship")
st.pyplot(box_plot1.get_figure())