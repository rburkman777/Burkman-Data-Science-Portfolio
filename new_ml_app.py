import streamlit as st

st.title("Machine Learning Streamlit App")

st.write("Welcome! Let's start by uploading your dataset! You can upload it as a CSV here:")

uploaded_file = st.file_uploader("Upload a CSV")

if uploaded_file:
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    st.write(df)

columns = df.columns.tolist()
    
    x_column = st.selectbox("Select feature (X)", columns)
    y_column = st.selectbox("Select target (y)", columns)

    if st.button("Train Model"):
        X = df[[x_column]]
        y = df[y_column]

        model = LinearRegression()
        model.fit(X, y)

        st.write("### Model Results")
        st.write(f"Coefficient: {model.coef_[0]}")
        st.write(f"Intercept: {model.intercept_}")

        # Predictions
        predictions = model.predict(X)

        df["Predictions"] = predictions
        st.write("### Predictions")
        st.write(df)
