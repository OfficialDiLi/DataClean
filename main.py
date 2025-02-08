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
    selected_column = st.selectbox("Select colunm to filter by", columns)
    unique_value = df[selected_column].unique()
    selected_value = st.selectbox("select value", unique_value)

    filtered_df = df[df[selected_column] == unique_value]
    st.write(filtered_df)

    st.subheader("Plot Data")
    x_column = st.selectbox("select the X-axis", columns)
    y_column = st.selectbox("select the Y-axis", columns)

    if st.button("Generate Plot"):
        st.line_chart(filtered_df.set_index(x_column)[y_column])
else:
    st.write("Waiting on file upload...")

