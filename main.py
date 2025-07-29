import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import missingno as msno
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from src.database.users import UserDB
from src.database.database import db_manager
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
)

st.set_page_config(page_title="DataClean Pro", page_icon="📊", layout="wide")


# --- Session State Initialization ---
def initialize_session_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "history" not in st.session_state:
        st.session_state.history = []
    if "df" not in st.session_state:
        st.session_state.df = None


initialize_session_state()


# --- Helper Functions ---
def update_state(df, message):
    st.session_state.history.append(df.copy())
    st.session_state.df = df
    st.toast(message, icon="✅")


def undo():
    if len(st.session_state.history) > 1:
        st.session_state.history.pop()
        st.session_state.df = st.session_state.history[-1]
        st.toast("Reverted to the previous state.", icon="↩️")
    else:
        st.warning("No previous state to undo.")


# --- Authentication UI ---
def show_login_signup():
    st.title("Welcome to DataClean Pro")
    menu = ["Login", "SignUp"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Login":
        st.subheader("Login Section")
        username = st.text_input("User Name")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            success, message = UserDB.validate_login(username, password)
            if success:
                st.session_state.logged_in = True
                st.success(f"Welcome {username}")
                st.rerun()
            else:
                st.warning(message)

    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        if st.button("Signup"):
            if UserDB.create_user(new_user, new_password):
                st.success("You have successfully created an account")
                st.info("Go to Login Menu to login")
            else:
                st.warning("Username already exists")


# --- Main Application UI ---
def show_main_app():
    st.title("DataClean Pro: Advanced Cleaning, Transformation, and Visualization")

    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader(
            "Choose a file", type=["csv", "xlsx", "xls", "ods"]
        )

        if uploaded_file is not None and st.session_state.df is None:
            file_extension = uploaded_file.name.split(".")[-1]
            if file_extension == "csv":
                initial_df = pd.read_csv(uploaded_file)
            elif file_extension in ["xlsx", "xls"]:
                initial_df = pd.read_excel(uploaded_file)
            elif file_extension == "ods":
                initial_df = pd.read_excel(uploaded_file, engine="odf")
            else:
                st.error("Unsupported file type.")
                initial_df = None

            if initial_df is not None:
                update_state(initial_df, "File uploaded successfully!")

        if st.session_state.df is not None:
            st.header("Actions")
            if st.button("Undo Last Action", use_container_width=True):
                undo()

            st.header("Export Data")
            csv = st.session_state.df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Cleaned Data as CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.df = None
            st.session_state.history = []
            st.rerun()

    if st.session_state.df is not None:
        df = st.session_state.df
        (
            tab_profiler,
            tab_view,
            tab_clean,
            tab_transform,
            tab_viz,
            tab_advanced,
            tab_ml,
            tab_group_insights,
            tab_merge,
        ) = st.tabs(
            [
                "Profiler",
                "Data View",
                "Cleaning",
                "Transformation",
                "Visualization",
                "Advanced Analysis",
                "ML Preprocessing",
                "Smart Group Insights",
                "Multi-File Merge",
            ]
        )

        with tab_profiler:
            st.header("Automated Data Profiling Report")
            if st.button("Generate Full Data Report"):
                pr = ProfileReport(df, title="DataClean Pro Report")
                st_profile_report(pr)

            st.subheader("One-Click Report Generator")
            if st.button("Generate and Download HTML Report"):
                with st.spinner("Generating report..."):
                    profile = ProfileReport(df, title="DataClean Pro Report")
                    report_html = profile.to_html()
                    st.download_button(
                        label="Download Report as HTML",
                        data=report_html,
                        file_name="data_profile_report.html",
                        mime="text/html",
                        use_container_width=True,
                    )
                st.success("Report generated and ready for download!")

        with tab_view:
            st.header("Current Data")
            st.dataframe(df)
            st.subheader("Data Summary")
            st.write(df.describe())

            st.subheader("Column ToolBox")
            selected_column_summary = st.selectbox(
                "Select a column for detailed summary",
                df.columns,
                key="column_summary_select",
            )
            if selected_column_summary:
                st.write(f"#### Summary for '{selected_column_summary}'")
                st.write(df[selected_column_summary].describe())
                if pd.api.types.is_numeric_dtype(df[selected_column_summary]):
                    fig_hist_summary = px.histogram(
                        df,
                        x=selected_column_summary,
                        title=f"Distribution of {selected_column_summary}",
                    )
                    st.plotly_chart(fig_hist_summary, use_container_width=True)
                elif pd.api.types.is_object_dtype(
                    df[selected_column_summary]
                ) or pd.api.types.is_categorical_dtype(df[selected_column_summary]):
                    value_counts = (
                        df[selected_column_summary].value_counts().reset_index()
                    )
                    value_counts.columns = ["Value", "Count"]
                    st.dataframe(value_counts)
                    fig_bar_summary = px.bar(
                        value_counts,
                        x="Value",
                        y="Count",
                        title=f"Value Counts of {selected_column_summary}",
                    )
                    st.plotly_chart(fig_bar_summary, use_container_width=True)

        with tab_clean:
            st.header("Cleaning Operations")
            with st.expander("Missing Data Heatmap"):
                fig, ax = plt.subplots()
                msno.matrix(df, ax=ax, sparkline=False)
                st.pyplot(fig)

            with st.expander("Change Data Types"):
                col_to_change = st.selectbox(
                    "Select column", df.columns, key="dtype_col"
                )
                new_type = st.selectbox(
                    "Select new type",
                    ["object (text)", "int64", "float64", "datetime64[ns]"],
                    key="dtype_new",
                )
                if st.button("Apply Type Change"):
                    try:
                        new_df = df.copy()
                        new_df[col_to_change] = new_df[col_to_change].astype(new_type)
                        update_state(
                            new_df, f"Changed type of '{col_to_change}' to {new_type}."
                        )
                    except Exception as e:
                        st.error(f"Failed to change type: {e}")

            with st.expander("Rename Columns"):
                col_to_rename = st.selectbox(
                    "Select column to rename", df.columns, key="rename_col"
                )
                new_name = st.text_input("Enter new column name", value=col_to_rename)
                if st.button("Rename Column"):
                    new_df = df.copy()
                    new_df.rename(columns={col_to_rename: new_name}, inplace=True)
                    update_state(new_df, f"Renamed '{col_to_rename}' to '{new_name}'.")

            with st.expander("Drop Columns"):
                cols_to_drop = st.multiselect(
                    "Select columns to drop", df.columns, key="drop_cols"
                )
                if st.button("Drop Selected Columns"):
                    new_df = df.drop(columns=cols_to_drop)
                    update_state(new_df, f"Dropped columns: {', '.join(cols_to_drop)}.")

            with st.expander("Handle Missing Values"):
                st.write("Missing values per column:", df.isnull().sum())
                if st.button("Drop all rows with missing values"):
                    new_df = df.dropna()
                    update_state(new_df, "Dropped rows with missing values.")

            with st.expander("Handle Duplicates"):
                st.write(f"Number of exact duplicate rows: {df.duplicated().sum()}")
                if st.button("Remove Exact Duplicate Rows"):
                    new_df = df.drop_duplicates()
                    update_state(new_df, "Removed exact duplicate rows.")

                st.subheader("Duplicate Detector Pro")
                duplicate_detection_type = st.radio(
                    "Select Duplicate Detection Type",
                    ["Column-level Duplicates", "Near-Duplicate Rows (Fuzzy Logic)"],
                )

                if duplicate_detection_type == "Column-level Duplicates":
                    col1_dup = st.selectbox(
                        "Select first column", df.columns, key="col1_dup"
                    )
                    col2_dup = st.selectbox(
                        "Select second column", df.columns, key="col2_dup"
                    )
                    if st.button("Detect Column-level Duplicates"):
                        if col1_dup and col2_dup:
                            if df[col1_dup].equals(df[col2_dup]):
                                st.success(
                                    f"Columns '{col1_dup}' and '{col2_dup}' are exact duplicates."
                                )
                            else:
                                st.info(
                                    f"Columns '{col1_dup}' and '{col2_dup}' are not exact duplicates."
                                )
                                # You could add more sophisticated similarity checks here if needed
                        else:
                            st.warning("Please select two columns.")

                elif duplicate_detection_type == "Near-Duplicate Rows (Fuzzy Logic)":
                    from rapidfuzz import fuzz

                    st.info(
                        "This feature detects near-duplicate rows based on fuzzy matching of selected text columns. It can be computationally intensive for large datasets."
                    )
                    fuzzy_cols = st.multiselect(
                        "Select text columns for fuzzy matching",
                        df.select_dtypes(include="object").columns.tolist(),
                        key="fuzzy_cols",
                    )
                    similarity_threshold = st.slider(
                        "Similarity Threshold (0-100)", 0, 100, 90
                    )

                    if st.button("Detect Near-Duplicate Rows") and fuzzy_cols:
                        with st.spinner(
                            "Detecting near-duplicate rows... This may take a while for large datasets."
                        ):
                            near_duplicates = []
                            # For simplicity, comparing each row to every other row.
                            # For larger datasets, consider more optimized approaches (e.g., blocking, LSH)
                            for i in range(len(df)):
                                for j in range(i + 1, len(df)):
                                    row1_str = " ".join(
                                        [str(df.iloc[i][col]) for col in fuzzy_cols]
                                    )
                                    row2_str = " ".join(
                                        [str(df.iloc[j][col]) for col in fuzzy_cols]
                                    )

                                    similarity = fuzz.ratio(row1_str, row2_str)
                                    if similarity >= similarity_threshold:
                                        near_duplicates.append((i, j, similarity))

                            if near_duplicates:
                                st.subheader("Detected Near-Duplicate Rows")
                                for dup in near_duplicates:
                                    st.write(
                                        f"Rows {dup[0]} and {dup[1]} are {dup[2]:.2f}% similar."
                                    )
                                    st.dataframe(df.iloc[[dup[0], dup[1]]])
                            else:
                                st.info(
                                    "No near-duplicate rows found based on the selected columns and threshold."
                                )
                    elif not fuzzy_cols and st.button("Detect Near-Duplicate Rows"):
                        st.warning(
                            "Please select at least one column for fuzzy matching."
                        )

            with st.expander("Outlier Detection Engine"):
                st.subheader("Detect and Handle Outliers (IQR Method)")
                outlier_col = st.selectbox(
                    "Select a numeric column for outlier detection",
                    df.select_dtypes(include=np.number).columns,
                    key="outlier_col",
                )

                if outlier_col:
                    Q1 = df[outlier_col].quantile(0.25)
                    Q3 = df[outlier_col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers_df = df[
                        (df[outlier_col] < lower_bound)
                        | (df[outlier_col] > upper_bound)
                    ]

                    st.write(f"**Column:** {outlier_col}")
                    st.write(f"**IQR:** {IQR:.2f}")
                    st.write(f"**Lower Bound:** {lower_bound:.2f}")
                    st.write(f"**Upper Bound:** {upper_bound:.2f}")
                    st.write(f"**Number of Outliers Detected:** {len(outliers_df)}")

                    if not outliers_df.empty:
                        st.dataframe(outliers_df)
                        fig_boxplot = px.box(
                            df,
                            y=outlier_col,
                            title=f"Boxplot of {outlier_col} with Outliers",
                        )
                        st.plotly_chart(fig_boxplot, use_container_width=True)

                        if st.button(
                            f"Remove {len(outliers_df)} Outliers from {outlier_col}"
                        ):
                            new_df = df[
                                (df[outlier_col] >= lower_bound)
                                & (df[outlier_col] <= upper_bound)
                            ]
                            update_state(
                                new_df,
                                f"Removed {len(outliers_df)} outliers from '{outlier_col}'.",
                            )
                    else:
                        st.info(
                            "No outliers detected in this column using the IQR method."
                        )

            with st.expander("Advanced Filtering"):
                filter_column = st.selectbox(
                    "Select column to filter", df.columns, key="filter_col"
                )
                if pd.api.types.is_numeric_dtype(df[filter_column]):
                    min_val, max_val = st.slider(
                        f"Filter {filter_column} by range",
                        float(df[filter_column].min()),
                        float(df[filter_column].max()),
                        (
                            float(df[filter_column].min()),
                            float(df[filter_column].max()),
                        ),
                    )
                    if st.button("Apply Numeric Filter"):
                        new_df = df[
                            (df[filter_column] >= min_val)
                            & (df[filter_column] <= max_val)
                        ]
                        update_state(
                            new_df,
                            f"Filtered {filter_column} between {min_val} and {max_val}.",
                        )
                else:
                    selected_values = st.multiselect(
                        f"Filter {filter_column} by values", df[filter_column].unique()
                    )
                    if st.button("Apply Categorical Filter"):
                        if selected_values:
                            new_df = df[df[filter_column].isin(selected_values)]
                            update_state(
                                new_df, f"Filtered {filter_column} to selected values."
                            )

        with tab_transform:
            st.header("Data Transformation")
            with st.expander("Bin Numeric Data"):
                bin_col = st.selectbox(
                    "Select numeric column to bin",
                    df.select_dtypes(include=np.number).columns,
                    key="bin_col",
                )
                bin_type = st.selectbox(
                    "Binning method", ["Equal-width (cut)", "Equal-frequency (qcut)"]
                )
                num_bins = st.slider("Number of bins", 2, 20, 5)
                if st.button("Apply Binning") and bin_col:
                    new_df = df.copy()
                    new_col_name = f"{bin_col}_binned"
                    if bin_type == "Equal-width (cut)":
                        new_df[new_col_name] = pd.cut(
                            new_df[bin_col],
                            bins=num_bins,
                            labels=False,
                            include_lowest=True,
                        )
                    else:
                        new_df[new_col_name] = pd.qcut(
                            new_df[bin_col], q=num_bins, labels=False, duplicates="drop"
                        )
                    update_state(new_df, f"Binned '{bin_col}' into {num_bins} bins.")

            with st.expander("Find and Replace"):
                replace_col = st.selectbox(
                    "Select column", df.columns, key="replace_col"
                )
                find_val = st.text_input("Value to find")
                replace_val = st.text_input("Value to replace with")
                if st.button("Apply Find and Replace"):
                    new_df = df.copy()
                    new_df[replace_col] = new_df[replace_col].replace(
                        find_val, replace_val
                    )
                    update_state(
                        new_df,
                        f"Replaced '{find_val}' with '{replace_val}' in '{replace_col}'.",
                    )

            with st.expander("Apply Single-Column Function"):
                transform_col = st.selectbox(
                    "Select column", df.columns, key="transform_col"
                )
                function = st.selectbox(
                    "Select function",
                    [
                        "log",
                        "sqrt",
                        "round",
                        "uppercase",
                        "lowercase",
                        "trim whitespace",
                    ],
                )
                if st.button("Apply Function"):
                    new_df = df.copy()
                    try:
                        if function == "log":
                            new_df[transform_col] = np.log(new_df[transform_col])
                        elif function == "sqrt":
                            new_df[transform_col] = np.sqrt(new_df[transform_col])
                        elif function == "round":
                            new_df[transform_col] = new_df[transform_col].round()
                        elif function == "uppercase":
                            new_df[transform_col] = new_df[transform_col].str.upper()
                        elif function == "lowercase":
                            new_df[transform_col] = new_df[transform_col].str.lower()
                        elif function == "trim whitespace":
                            new_df[transform_col] = new_df[transform_col].str.strip()
                        update_state(
                            new_df, f"Applied {function} to '{transform_col}'."
                        )
                    except Exception as e:
                        st.error(f"Could not apply function: {e}")

            with st.expander("Create New Column from Two Columns"):
                col1 = st.selectbox("Select first column", df.columns, key="col1")
                col2 = st.selectbox("Select second column", df.columns, key="col2")
                operation = st.selectbox("Select operation", ["+", "-", "*", "/"])
                new_col_name = st.text_input("Enter new column name")
                if st.button("Create Column"):
                    if new_col_name:
                        new_df = df.copy()
                        try:
                            if operation == "+":
                                new_df[new_col_name] = new_df[col1] + new_df[col2]
                            elif operation == "-":
                                new_df[new_col_name] = new_df[col1] - new_df[col2]
                            elif operation == "*":
                                new_df[new_col_name] = new_df[col1] * new_df[col2]
                            elif operation == "/":
                                new_df[new_col_name] = new_df[col1] / new_df[col2]
                            update_state(
                                new_df, f"Created new column '{new_col_name}'."
                            )
                        except Exception as e:
                            st.error(f"Could not perform operation: {e}")
                    else:
                        st.warning("Please enter a name for the new column.")

            with st.expander("Extract Parts from Text Column"):
                extract_col = st.selectbox(
                    "Select text column to extract from",
                    df.select_dtypes(include="object").columns,
                    key="extract_col",
                )
                extract_method = st.selectbox(
                    "Select extraction method",
                    ["First N Characters", "Split by Delimiter"],
                )

                if extract_method == "First N Characters":
                    num_chars = st.number_input(
                        "Number of characters to extract", min_value=1, value=5
                    )
                    if st.button("Extract First N Characters"):
                        new_df = df.copy()
                        new_df[f"{extract_col}_first_{num_chars}"] = (
                            new_df[extract_col].astype(str).str[:num_chars]
                        )
                        update_state(
                            new_df,
                            f"Extracted first {num_chars} characters from '{extract_col}'.",
                        )
                elif extract_method == "Split by Delimiter":
                    delimiter = st.text_input("Enter delimiter (e.g., comma, space)")
                    split_index = st.number_input(
                        "Index of part to extract (0 for first, 1 for second, etc.)",
                        min_value=0,
                        value=0,
                    )
                    if st.button("Split and Extract"):
                        if delimiter:
                            new_df = df.copy()
                            try:
                                new_df[f"{extract_col}_part_{split_index}"] = (
                                    new_df[extract_col]
                                    .astype(str)
                                    .str.split(delimiter)
                                    .str[split_index]
                                )
                                update_state(
                                    new_df,
                                    f"Extracted part {split_index} from '{extract_col}' by splitting with '{delimiter}'.",
                                )
                            except Exception as e:
                                st.error(f"Could not split and extract: {e}")
                        else:
                            st.warning("Please enter a delimiter.")

        with tab_viz:
            st.header("Data Visualization")

            with st.expander("Interactive Plotting", expanded=True):
                plot_type = st.selectbox(
                    "Select plot type",
                    [
                        "scatter",
                        "line",
                        "bar",
                        "histogram",
                        "box",
                        "sunburst",
                        "treemap",
                    ],
                )
                if plot_type in ["scatter", "line", "bar", "box"]:
                    x_axis = st.selectbox("Select X-axis", df.columns, key="x_axis")
                    y_axis = st.selectbox("Select Y-axis", df.columns, key="y_axis")
                    fig = go.Figure()
                    if plot_type == "scatter":
                        fig.add_trace(
                            go.Scatter(x=df[x_axis], y=df[y_axis], mode="markers")
                        )
                    elif plot_type == "line":
                        fig.add_trace(
                            go.Scatter(x=df[x_axis], y=df[y_axis], mode="lines")
                        )
                    elif plot_type == "bar":
                        fig.add_trace(go.Bar(x=df[x_axis], y=df[y_axis]))
                    elif plot_type == "box":
                        fig.add_trace(go.Box(y=df[y_axis], x=df[x_axis]))
                    st.plotly_chart(fig, use_container_width=True)
                elif plot_type == "histogram":
                    hist_col = st.selectbox(
                        "Select column for histogram", df.columns, key="hist_col"
                    )
                    fig = px.histogram(df, x=hist_col)
                    st.plotly_chart(fig, use_container_width=True)
                elif plot_type == "sunburst":
                    path_cols = st.multiselect(
                        "Select path columns (hierarchy)",
                        df.columns,
                        key="sunburst_path",
                    )
                    value_col = st.selectbox(
                        "Select value column",
                        df.select_dtypes(include=np.number).columns,
                        key="sunburst_val",
                    )
                    if st.button("Generate Sunburst Chart") and path_cols and value_col:
                        fig = px.sunburst(df, path=path_cols, values=value_col)
                        st.plotly_chart(fig, use_container_width=True)
                elif plot_type == "treemap":
                    path_cols_tree = st.multiselect(
                        "Select path columns (hierarchy)", df.columns, key="tree_path"
                    )
                    value_col_tree = st.selectbox(
                        "Select value column",
                        df.select_dtypes(include=np.number).columns,
                        key="tree_val",
                    )
                    if (
                        st.button("Generate Treemap")
                        and path_cols_tree
                        and value_col_tree
                    ):
                        fig = px.treemap(df, path=path_cols_tree, values=value_col_tree)
                        st.plotly_chart(fig, use_container_width=True)

            with st.expander("Correlation Heatmap"):
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if len(numeric_cols) > 1:
                    corr = df[numeric_cols].corr()
                    fig = px.imshow(corr, text_auto=True, aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(
                        "Not enough numeric columns to generate a correlation heatmap."
                    )

            with st.expander("Data Timeline (Time-Series Enhancement)"):
                datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
                if datetime_cols:
                    time_col = st.selectbox(
                        "Select a datetime column", datetime_cols, key="time_col"
                    )
                    value_col_ts = st.selectbox(
                        "Select a numeric column for analysis",
                        df.select_dtypes(include=np.number).columns,
                        key="value_col_ts",
                    )

                    if time_col and value_col_ts:
                        st.subheader(
                            f"Time-Series Analysis for {value_col_ts} over {time_col}"
                        )

                        # Line plot over time
                        fig_line_ts = px.line(
                            df.sort_values(by=time_col),
                            x=time_col,
                            y=value_col_ts,
                            title=f"Time Series of {value_col_ts}",
                        )
                        st.plotly_chart(fig_line_ts, use_container_width=True)

                        # Rolling Averages
                        window_size = st.slider(
                            "Rolling Average Window Size",
                            2,
                            30,
                            7,
                            key="rolling_window",
                        )
                        df_temp_ts = df.copy().sort_values(by=time_col)
                        df_temp_ts[f"{value_col_ts}_rolling_mean"] = (
                            df_temp_ts[value_col_ts].rolling(window=window_size).mean()
                        )

                        fig_rolling = px.line(
                            df_temp_ts,
                            x=time_col,
                            y=[value_col_ts, f"{value_col_ts}_rolling_mean"],
                            title=f"Time Series with {window_size}-day Rolling Mean",
                        )
                        st.plotly_chart(fig_rolling, use_container_width=True)

                        # Percentage Change
                        if st.checkbox("Show Percentage Change"):
                            df_temp_ts[f"{value_col_ts}_pct_change"] = (
                                df_temp_ts[value_col_ts].pct_change() * 100
                            )
                            fig_pct_change = px.line(
                                df_temp_ts,
                                x=time_col,
                                y=f"{value_col_ts}_pct_change",
                                title=f"Percentage Change of {value_col_ts}",
                            )
                            st.plotly_chart(fig_pct_change, use_container_width=True)

                        # Seasonality (simplified - requires more advanced analysis for true seasonality)
                        if st.checkbox("Show Monthly Average (Simplified Seasonality)"):
                            df_temp_ts["month"] = df_temp_ts[time_col].dt.month
                            monthly_avg = (
                                df_temp_ts.groupby("month")[value_col_ts]
                                .mean()
                                .reset_index()
                            )
                            fig_monthly = px.bar(
                                monthly_avg,
                                x="month",
                                y=value_col_ts,
                                title=f"Monthly Average of {value_col_ts}",
                            )
                            st.plotly_chart(fig_monthly, use_container_width=True)

                else:
                    st.info(
                        "No datetime columns found in the dataset for time-series analysis."
                    )

        with tab_advanced:
            st.header("Advanced Analysis")
            with st.expander("Pivot Table Generator"):
                index_col = st.multiselect(
                    "Select index columns", df.columns, key="pivot_index"
                )
                col_col = st.multiselect(
                    "Select columns for new columns", df.columns, key="pivot_cols"
                )
                val_col = st.selectbox(
                    "Select value column",
                    df.select_dtypes(include=np.number).columns,
                    key="pivot_val",
                )
                agg_func = st.selectbox(
                    "Select aggregation function",
                    ["sum", "mean", "count", "min", "max"],
                    key="pivot_agg",
                )
                if st.button("Generate Pivot Table") and index_col and val_col:
                    try:
                        pivot_df = df.pivot_table(
                            index=index_col,
                            columns=col_col,
                            values=val_col,
                            aggfunc=agg_func,
                        )
                        st.dataframe(pivot_df)
                    except Exception as e:
                        st.error(f"Could not create pivot table: {e}")

            with st.expander("SQL-like Query"):
                query_str = st.text_area(
                    "Enter your query (e.g., Age > 30 and Gender == 'Male')"
                )
                if st.button("Run Query") and query_str:
                    try:
                        queried_df = df.query(query_str)
                        st.dataframe(queried_df)
                        if st.checkbox("Update main dataframe with query result?"):
                            update_state(
                                queried_df, f"Dataframe updated with query: {query_str}"
                            )
                    except Exception as e:
                        st.error(f"Invalid query: {e}")

        with tab_ml:
            st.header("Machine Learning Preprocessing")
            st.subheader("Feature Scaling (for Numeric Columns)")
            scale_col = st.selectbox(
                "Select numeric column to scale",
                df.select_dtypes(include=np.number).columns,
                key="scale_col",
            )
            scaler_type = st.selectbox(
                "Select scaler", ["StandardScaler", "MinMaxScaler"]
            )
            if st.button("Apply Scaler") and scale_col:
                new_df = df.copy()
                scaler = (
                    StandardScaler()
                    if scaler_type == "StandardScaler"
                    else MinMaxScaler()
                )
                new_df[[scale_col]] = scaler.fit_transform(new_df[[scale_col]])
                update_state(new_df, f"Applied {scaler_type} to '{scale_col}'.")

            st.subheader("Handle Missing Values")
            missing_col = st.selectbox(
                "Select column to handle missing values", df.columns, key="missing_col"
            )
            if missing_col:
                missing_strategy = st.selectbox(
                    "Select strategy",
                    [
                        "Drop Rows",
                        "Impute with Mean (Numeric)",
                        "Impute with Median (Numeric)",
                        "Impute with Mode (Categorical/Numeric)",
                    ],
                )

                if st.button("Apply Missing Value Strategy"):
                    new_df = df.copy()
                    if missing_strategy == "Drop Rows":
                        new_df.dropna(subset=[missing_col], inplace=True)
                        update_state(
                            new_df,
                            f"Dropped rows with missing values in '{missing_col}'.",
                        )
                    elif missing_strategy == "Impute with Mean (Numeric)":
                        if pd.api.types.is_numeric_dtype(new_df[missing_col]):
                            new_df[missing_col].fillna(
                                new_df[missing_col].mean(), inplace=True
                            )
                            update_state(
                                new_df,
                                f"Imputed missing values in '{missing_col}' with mean.",
                            )
                        else:
                            st.warning(
                                "Selected column is not numeric for mean imputation."
                            )
                    elif missing_strategy == "Impute with Median (Numeric)":
                        if pd.api.types.is_numeric_dtype(new_df[missing_col]):
                            new_df[missing_col].fillna(
                                new_df[missing_col].median(), inplace=True
                            )
                            update_state(
                                new_df,
                                f"Imputed missing values in '{missing_col}' with median.",
                            )
                        else:
                            st.warning(
                                "Selected column is not numeric for median imputation."
                            )
                    elif missing_strategy == "Impute with Mode (Categorical/Numeric)":
                        new_df[missing_col].fillna(
                            new_df[missing_col].mode()[0], inplace=True
                        )
                        update_state(
                            new_df,
                            f"Imputed missing values in '{missing_col}' with mode.",
                        )

            st.subheader("Categorical Variable Encoding")
            encode_col = st.selectbox(
                "Select categorical column to encode",
                df.select_dtypes(include="object").columns,
                key="encode_col",
            )
            encoder_type = st.selectbox(
                "Select encoder", ["LabelEncoder", "OneHotEncoder"]
            )
            if st.button("Apply Encoder") and encode_col:
                new_df = df.copy()
                if encoder_type == "LabelEncoder":
                    le = LabelEncoder()
                    new_df[encode_col] = le.fit_transform(new_df[encode_col])
                    update_state(new_df, f"Applied LabelEncoder to '{encode_col}'.")
                else:
                    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                    encoded_cols = pd.DataFrame(ohe.fit_transform(new_df[[encode_col]]))
                    encoded_cols.columns = ohe.get_feature_names_out([encode_col])
                    new_df = pd.concat(
                        [new_df.drop(columns=[encode_col]), encoded_cols], axis=1
                    )
                    update_state(new_df, f"Applied OneHotEncoder to '{encode_col}'.")

            st.subheader("Target Variable Inspector")
            target_variable = st.selectbox(
                "Select Target Variable", df.columns, key="target_var_inspector"
            )

            if target_variable:
                st.write(f"### Analysis for Target Variable: {target_variable}")

                # Correlation with numeric features
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

                if pd.api.types.is_numeric_dtype(df[target_variable]):
                    if target_variable in numeric_cols:
                        numeric_cols.remove(
                            target_variable
                        )  # Remove target variable itself from features

                    if numeric_cols:
                        st.write("#### Correlation with Numeric Features")
                        correlations = (
                            df[numeric_cols]
                            .corrwith(df[target_variable])
                            .sort_values(ascending=False)
                        )
                        st.dataframe(correlations.to_frame(name="Correlation"))
                    else:
                        st.info("No other numeric columns to calculate correlation.")
                else:
                    st.info(
                        "Correlation with numeric features can only be calculated for a numeric target variable."
                    )

                # Class Imbalance (for classification)
                if (
                    pd.api.types.is_categorical_dtype(df[target_variable])
                    or df[target_variable].nunique() < 20
                ):  # Heuristic for categorical
                    st.write("#### Class Imbalance")
                    class_counts = df[target_variable].value_counts()
                    st.dataframe(class_counts.to_frame(name="Count"))
                    fig_pie = px.pie(
                        class_counts,
                        values=class_counts.values,
                        names=class_counts.index,
                        title=f"Class Distribution of {target_variable}",
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info(
                        "Target variable is not categorical or has too many unique values for class imbalance analysis."
                    )

                # Plot stacked bar/violin plots per target category
                st.write("#### Feature Distribution per Target Category")
                feature_to_plot = st.selectbox(
                    "Select a feature to plot against the target",
                    df.columns.drop(target_variable).tolist(),
                    key="feature_plot_target",
                )

                if feature_to_plot:
                    if pd.api.types.is_numeric_dtype(df[feature_to_plot]):
                        fig_violin = px.violin(
                            df,
                            x=target_variable,
                            y=feature_to_plot,
                            box=True,
                            points="all",
                            title=f"Distribution of {feature_to_plot} by {target_variable}",
                        )
                        st.plotly_chart(fig_violin, use_container_width=True)
                    elif (
                        pd.api.types.is_categorical_dtype(df[feature_to_plot])
                        or df[feature_to_plot].nunique() < 20
                    ):
                        # Stacked bar chart for two categorical variables
                        temp_df = (
                            df.groupby([target_variable, feature_to_plot])
                            .size()
                            .reset_index(name="count")
                        )
                        fig_stacked_bar = px.bar(
                            temp_df,
                            x=target_variable,
                            y="count",
                            color=feature_to_plot,
                            title=f"Stacked Bar of {feature_to_plot} by {target_variable}",
                        )
                        st.plotly_chart(fig_stacked_bar, use_container_width=True)
                    else:
                        st.info(
                            "Selected feature is not suitable for plotting against the target variable (e.g., too many unique values)."
                        )
            else:
                st.info("Please select a target variable to inspect.")

        with tab_merge:
            st.header("Multi-File Merge")
            st.write("Upload multiple CSV files and merge them.")

            uploaded_files_merge = st.file_uploader(
                "Choose CSV files to merge",
                type="csv",
                accept_multiple_files=True,
                key="merge_files",
            )

            if uploaded_files_merge:
                dataframes_to_merge = {}
                for uploaded_file_merge in uploaded_files_merge:
                    df_name = uploaded_file_merge.name.split(".")[0]
                    dataframes_to_merge[df_name] = pd.read_csv(uploaded_file_merge)

                # Store dataframes in session state for persistence
                st.session_state.dataframes_to_merge = dataframes_to_merge

                df_names = list(dataframes_to_merge.keys())
                if len(df_names) >= 2:
                    st.subheader("Select DataFrames for Merging")
                    primary_df_name = st.selectbox(
                        "Select Primary DataFrame", df_names, key="primary_df_name"
                    )
                    secondary_df_name = st.selectbox(
                        "Select Secondary DataFrame",
                        [name for name in df_names if name != primary_df_name],
                        key="secondary_df_name",
                    )

                    if primary_df_name and secondary_df_name:
                        df1_merge = dataframes_to_merge[primary_df_name]
                        df2_merge = dataframes_to_merge[secondary_df_name]

                        st.write(
                            f"**Primary DataFrame ({primary_df_name}) Columns:** {df1_merge.columns.tolist()}"
                        )
                        st.write(
                            f"**Secondary DataFrame ({secondary_df_name}) Columns:** {df2_merge.columns.tolist()}"
                        )

                        common_cols = list(
                            set(df1_merge.columns) & set(df2_merge.columns)
                        )
                        merge_on = st.multiselect(
                            "Select column(s) to merge on (common columns)",
                            common_cols,
                            key="merge_on",
                        )
                        merge_how = st.selectbox(
                            "Select Merge Type",
                            ["inner", "outer", "left", "right"],
                            key="merge_how",
                        )

                        if st.button("Perform Merge"):
                            if merge_on:
                                try:
                                    merged_df = pd.merge(
                                        df1_merge, df2_merge, on=merge_on, how=merge_how
                                    )
                                    update_state(
                                        merged_df,
                                        f"Successfully merged {primary_df_name} and {secondary_df_name} with a {merge_how} join on {merge_on}.",
                                    )
                                    st.subheader("Merged Data Preview")
                                    st.dataframe(merged_df)
                                except Exception as e:
                                    st.error(f"Error during merge: {e}")
                            else:
                                st.warning(
                                    "Please select at least one column to merge on."
                                )
                    else:
                        st.info("Please select two different dataframes to merge.")
                else:
                    st.info("Upload at least two CSV files to enable merging.")

        with tab_group_insights:
            st.header("Smart Group Insights")
            st.write(
                "Select a categorical column and a numerical column to generate grouped statistics and charts."
            )

            categorical_cols = df.select_dtypes(include="object").columns.tolist()
            numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

            if categorical_cols and numerical_cols:
                group_col = st.selectbox(
                    "Select Grouping Column (Categorical)",
                    categorical_cols,
                    key="group_col_insights",
                )
                target_col = st.selectbox(
                    "Select Target Column (Numerical)",
                    numerical_cols,
                    key="target_col_insights",
                )

                if st.button("Generate Group Insights"):
                    if group_col and target_col:
                        grouped_df = (
                            df.groupby(group_col)[target_col]
                            .agg(["count", "mean", "median", "min", "max"])
                            .reset_index()
                        )
                        st.subheader(
                            f"Grouped Statistics by {group_col} for {target_col}"
                        )
                        st.dataframe(grouped_df)

                        fig = px.bar(
                            grouped_df,
                            x=group_col,
                            y="mean",
                            title=f"Mean {target_col} by {group_col}",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Please select both a grouping and a target column.")
            else:
                st.info(
                    "Please ensure your dataset has both categorical and numerical columns to use this feature."
                )

    else:
        st.info("Upload a CSV file using the sidebar to get started.")


if st.session_state.logged_in:
    show_main_app()
else:
    show_login_signup()
