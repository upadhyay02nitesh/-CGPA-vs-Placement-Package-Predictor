import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Set style
sns.set(style="whitegrid")

def load_data(file):  
    try:
        data = pd.read_csv(file)
        st.success("✅ Data loaded successfully.")
        return data
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

def clean_data(data):
    if data is None:
        st.warning("⚠️ No data to clean.")
        return None

    st.subheader("🧹 Cleaning Data")

    original_rows = data.shape[0]

    # Show duplicates
    duplicates = data[data.duplicated()]
    st.markdown(f"**Duplicate Rows:** {duplicates.shape[0]}")
    st.dataframe(duplicates)

    duplicate_rows = data[data.duplicated(keep=False)]
    st.write("**Missing Values:**",data.duplicated().sum())
    
    st.write("### 🔍 Duplicate Rows Heatmap")
    st.markdown(
        """
        This heatmap visualizes the **structure of duplicated rows** in your dataset.
        
        - **Lighter colors** indicate higher values (1), meaning presence of data.
        - **Darker colors** (0) indicate absence or lower values (if any).
        
        This helps visually inspect the distribution of values in duplicated rows.
        """
    )

    fig, ax = plt.subplots()
    sns.heatmap(duplicate_rows.astype(int), cmap="YlGnBu", cbar=False, ax=ax)
    st.pyplot(fig)



    # Drop duplicates
    data = data.drop_duplicates().reset_index(drop=True)
    after_drop_rows = data.shape[0]

    st.markdown(f"✅ Removed {original_rows - after_drop_rows} duplicate rows.")


    st.write("** AFter Remove Duplicate Values:**",data.duplicated().sum())
    duplicates = data[data.duplicated()]
    st.markdown(f"**Duplicate Rows:** {duplicates.shape[0]}")
    st.dataframe(duplicates)

    st.write("### 🔍 Duplicate Rows Heatmap")
    st.markdown(
        """
        This heatmap visualizes the **structure of duplicated rows** in your dataset.
        
        - **Lighter colors** indicate higher values (1), meaning presence of data.
        - **Darker colors** (0) indicate absence or lower values (if any).
        
        This helps visually inspect the distribution of values in duplicated rows.
        """
    )

    fig, ax = plt.subplots()
    sns.heatmap(data=data, cmap="YlGnBu", cbar=False, ax=ax)
    st.pyplot(fig)
    # Show missing values
    st.write("**Missing Values:**",data.isna().sum())
    
    st.write("### 🔍 Missing Values Heatmap")
    st.markdown("This heatmap shows the presence of missing values in the dataset.")
    st.markdown("**Yellow** indicates missing values.")
    st.markdown("**Purple** indicates no missing values.")
    fig, ax = plt.subplots()
    sns.heatmap(data.isna(), cbar=False, cmap="viridis", ax=ax)
    st.pyplot(fig)

    null_rows = data[data.isnull().any(axis=1)]
    st.markdown(f"**Rows with Missing Values:** {null_rows.shape[0]}")
    st.dataframe(null_rows)

    # Fill missing values in numeric columns with mean
    for column in data.select_dtypes(include=[np.number]).columns:
        data[column].fillna(data[column].mean(), inplace=True)
    # Reset index for consistency

    st.write("### 🔍After Remove Missing Values Heatmap")
    st.markdown("This heatmap shows the presence of missing values in the dataset.")
    st.markdown("**Yellow** indicates missing values.")
    st.markdown("**Purple** indicates no missing values.")
    fig, ax = plt.subplots()
    sns.heatmap(data.isna(), cbar=False, cmap="viridis", ax=ax)
    st.pyplot(fig)

    st.success("✅ Data cleaned successfully.")
    st.markdown(f"📦 Final row count: {data.shape[0]}")
    return data

def visualize_data(data):
    if data is None:
        st.warning("⚠️ No data to visualize.")
        return

    if 'CGPA' in data.columns and 'Package(LPA)' in data.columns:
        st.markdown("### Scatter Plot: CGPA vs Package")
        fig, ax = plt.subplots()
        sns.regplot(data=data, x="CGPA", y="Package(LPA)", ax=ax, line_kws={"color": "red"})

        ax.set_xlabel("CGPA")
        ax.set_ylabel("Package (LPA)")
        st.pyplot(fig)
    else:
        st.warning("CGPA or Package(LPA) column not found for scatter plot.")
def detect_outliers_iqr(data):
    st.subheader("📊 Outlier Detection Using IQR Method")

    numeric_cols = data.select_dtypes(include='number').columns
    outlier_indices = set()

    st.write("### 📦 Box Plot: Package Distribution by CGPA (with Outliers)")

    
    

    # Calculate IQR bounds for CGPA
    # CGPA Outlier Thresholds using IQR
    Q1_cgpa = data['CGPA'].quantile(0.25)
    Q3_cgpa = data['CGPA'].quantile(0.75)
    IQR_cgpa = Q3_cgpa - Q1_cgpa
    lower_cgpa = Q1_cgpa - 1.5 * IQR_cgpa
    upper_cgpa = Q3_cgpa + 1.5 * IQR_cgpa

    # Package Outlier Thresholds using IQR
    Q1_package = data['Package(LPA)'].quantile(0.25)
    Q3_package = data['Package(LPA)'].quantile(0.75)
    IQR_package = Q3_package - Q1_package
    lower_package = Q1_package - 1.5 * IQR_package
    upper_package = Q3_package + 1.5 * IQR_package




    # Outliers in either CGPA or Package
    outliers = data[
        (data["CGPA"] < lower_cgpa) | (data["CGPA"] > upper_cgpa) |
        (data["Package(LPA)"] < lower_package) | (data["Package(LPA)"] > upper_package)
    ]
    st.write(len(outliers), "outliers detected.")
    st.write(outliers, "outliers detected.")
    # Non-outliers (must be inside bounds for both CGPA and Package)
    non_outliers = data[
        (data["CGPA"] >= lower_cgpa) & (data["CGPA"] <= upper_cgpa) &
        (data["Package(LPA)"] >= lower_package) & (data["Package(LPA)"] <= upper_package)
    ]

    st.write("### 🎯 Scatter Plot: CGPA vs Package (Outliers Highlighted)")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=non_outliers, x='CGPA', y='Package(LPA)', label='Normal', color='blue', ax=ax)
    sns.scatterplot(data=outliers, x='CGPA', y='Package(LPA)', label='Outliers', color='red', marker='X', s=100, ax=ax)

    ax.set_title('CGPA vs Package (Outliers Highlighted)')
    ax.set_xlabel('CGPA')
    ax.set_ylabel('Package (LPA)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    st.pyplot(fig)






    # Filter rows where both CGPA and Package are within bounds
    filtered_data = data[
        (data['CGPA'] >= lower_cgpa) & (data['CGPA'] <= upper_cgpa) &
        (data['Package(LPA)'] >= lower_package) & (data['Package(LPA)'] <= upper_package)
    ]


    st.write("### 📦 Filtered Data (Outliers Removed)")
    outliers = filtered_data[
        (data["CGPA"] < lower_cgpa) | (data["CGPA"] > upper_cgpa) |
        (data["Package(LPA)"] < lower_package) | (data["Package(LPA)"] > upper_package)
    ]
    st.write(len(outliers), "outliers detected.")
    st.write(outliers, "outliers detected.")
    # Non-outliers (must be inside bounds for both CGPA and Package)
    non_outliers = filtered_data[
        (filtered_data["CGPA"] >= lower_cgpa) & (filtered_data["CGPA"] <= upper_cgpa) &
        (filtered_data["Package(LPA)"] >= lower_package) & (filtered_data["Package(LPA)"] <= upper_package)
    ]

    st.write("### 🎯 Scatter Plot: CGPA vs Package (Outliers Highlighted)")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=non_outliers, x='CGPA', y='Package(LPA)', label='Normal', color='blue', ax=ax)
    sns.scatterplot(data=outliers, x='CGPA', y='Package(LPA)', label='Outliers', color='red', marker='X', s=100, ax=ax)

    ax.set_title('CGPA vs Package (Outliers Highlighted)')
    ax.set_xlabel('CGPA')
    ax.set_ylabel('Package (LPA)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    st.pyplot(fig)
    return filtered_data

st.title("📈 Student CGPA & Package Analysis App")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = load_data(uploaded_file)

    if st.checkbox("🔍 Show Raw Data"):
        st.dataframe(data)

    cleaned_data = clean_data(data)
    if st.checkbox("🔍 Show Cleaned Data"):
        st.dataframe(cleaned_data)
        st.write(f"Original data rows: {data.shape[0]}")
        st.write(f"Cleaned data rows: {cleaned_data.shape[0]}")

        st.write("Missing values before cleaning:")
        st.write(data.isnull().sum())

        st.write("Missing values after cleaning:")
        st.write(cleaned_data.isnull().sum())

    if st.checkbox("📊 Show Visualizations"):
        visualize_data(cleaned_data)
    if st.checkbox("🚫 Detect and Remove Outliers"):
        data_no_outliers = detect_outliers_iqr(cleaned_data)
        st.write("### ✅ Data after removing outliers:")
        st.dataframe(data_no_outliers)
        csv = data_no_outliers.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Cleaned Data as CSV",
            data=csv,
            file_name='cleaned_data.csv',
            mime='text/csv'
        )
else:
    st.info("👆 Upload a CSV file to begin.")