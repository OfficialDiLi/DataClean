import streamlit as st
import pandas as pd

st.title("Simple Data Analysis App")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Sample")
    st.write(df.head())

    st.subheader("Data Summary")
    st.write(df.describe())

    st.subheader("Filter Data")
    columns = df.columns.tolist()
    selected_column = st.selectbox("Select column to filter by", columns)
    unique_values = df[selected_column].unique()
    selected_value = st.selectbox("Select value", unique_values)

    # Corrected line: replaced 'unique_value' with 'selected_value'
    filtered_df = df[df[selected_column] == selected_value]
    st.write(filtered_df)

    st.subheader("Plot Data")
    x_column = st.selectbox("Select the X-axis", columns)
    y_column = st.selectbox("Select the Y-axis", columns)

    if st.button("Generate Plot"):
        st.line_chart(data=filtered_df, x=x_column, y=y_column)
else:
    st.write("Waiting on file upload...")
