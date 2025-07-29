# DataClean Pro: Your Advanced Data Cleaning and Analysis Co-Pilot

DataClean Pro is a powerful and intuitive Streamlit-based web application designed to streamline your entire data workflow. From cleaning and transformation to advanced analysis and visualization, this tool provides a comprehensive suite of features in a user-friendly interface, making it an indispensable co-pilot for data professionals and enthusiasts alike.

## Key Features

- **Seamless Data Ingestion**: Upload datasets in various formats, including CSV, Excel (.xlsx, .xls), and LibreOffice Calc (.ods).
- **Secure User Authentication**: Protect your work with a secure login/signup system.
- **Interactive Data Profiling**: Generate a comprehensive `ydata-profiling` report to quickly understand your dataset’s characteristics, including missing values, correlations, and distributions.
- **Advanced Cleaning Operations**:
    - Visualize missing data with a heatmap.
    - Handle duplicates with both exact and fuzzy matching.
    - Detect and remove outliers using the IQR method.
    - Filter data with SQL-like queries.
- **Powerful Transformation Tools**:
    - Change data types, rename columns, and apply functions.
    - Create new features from existing columns.
    - Bin numeric data and extract text parts with ease.
- **Rich Visualization Suite**:
    - Generate interactive plots (scatter, line, bar, histogram, box).
    - Create hierarchical charts like sunbursts and treemaps.
    - Analyze time-series data with rolling averages and seasonality plots.
- **ML Preprocessing**:
    - Scale numeric features using `StandardScaler` or `MinMaxScaler`.
    - Encode categorical variables with `LabelEncoder` or `OneHotEncoder`.
    - Inspect your target variable with correlation analysis and class imbalance reports.
- **Undo/Redo Functionality**: Never worry about making a mistake. Every action is recorded, allowing you to revert to a previous state at any time.
- **Export and Share**: Download your cleaned data or the full HTML report to share your findings.

## How to Set Up on Your Local Machine

Get DataClean Pro up and running on your local machine in just a few steps.

### Prerequisites

- **Python 3.8+**
- **`uv` Python Package Installer**: This project uses `uv` for fast and efficient dependency management. If you don't have it, install it with:
  ```bash
  pip install uv
  ```

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/DataClean.git
   cd DataClean
   ```
   *(Replace `your-username` with the actual repository owner's username)*

2. **Create a Virtual Environment and Install Dependencies**:
   `uv` will create a virtual environment and install all required packages from `requirement.txt`.
   ```bash
   uv venv
   uv pip install -r requirement.txt
   ```

3. **Activate the Virtual Environment**:
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```
   - **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```

4. **Run the Application**:
   Launch the Streamlit application with the following command:
   ```bash
   streamlit run main.py
   ```
   The application will open in your default web browser.

## How to Use DataClean Pro

1. **Sign Up/Login**: Create a new account or log in to get started.
2. **Upload Your Data**: In the sidebar, upload your dataset in any of the supported formats.
3. **Explore and Profile**:
   - Use the **Profiler** tab to generate a detailed report of your data.
   - View the raw data and get a quick summary in the **Data View** tab.
4. **Clean and Transform**:
   - Navigate through the **Cleaning** and **Transformation** tabs to apply various operations.
   - Use the **Undo** button in the sidebar if you need to revert any changes.
5. **Visualize and Analyze**:
   - Create custom plots in the **Visualization** tab.
   - Use the **Advanced Analysis** and **Smart Group Insights** tabs to uncover deeper insights.
6. **Export Your Results**:
   - Download the cleaned dataset as a CSV file.
   - Download the full data profile as an HTML report.

Enjoy a seamless and powerful data analysis experience with DataClean Pro!