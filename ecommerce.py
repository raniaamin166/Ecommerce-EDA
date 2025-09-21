import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# -------------------
# STREAMLIT APP
# -------------------
st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")
st.title("🛒 E-Commerce Data Analysis Dashboard")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Sidebar navigation
    menu = st.sidebar.radio("Select Section", [
        "📊 Overview",
        "📈 Column Analysis",
        "📉 Correlation & Regression",
        "🔥 Advanced Business Analysis"
    ])

    # -------------------
    # 1. Overview
    # -------------------
    if menu == "📊 Overview":
        st.subheader("Dataset Preview")
        st.write(df.head())

        st.subheader("Summary Statistics")
        st.write(df.describe(include="all"))

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

    # -------------------
    # 2. Column Analysis
    # -------------------
    elif menu == "📈 Column Analysis":
        column = st.selectbox("Select a column for analysis", df.columns)

        if pd.api.types.is_numeric_dtype(df[column]):
            # Histogram
            fig, ax = plt.subplots()
            sns.histplot(df[column].dropna(), bins=20, color="skyblue", ax=ax)
            ax.set_title(f"Histogram of {column}")
            st.pyplot(fig)

            # Boxplot
            fig, ax = plt.subplots()
            sns.boxplot(y=df[column], color="lightgreen", ax=ax)
            ax.set_title(f"Boxplot of {column}")
            st.pyplot(fig)

        else:
            # Bar chart
            fig, ax = plt.subplots()
            df[column].value_counts().plot(kind="bar", ax=ax, color="coral")
            ax.set_title(f"Bar Chart of {column}")
            st.pyplot(fig)

            # Pie chart
            fig, ax = plt.subplots()
            df[column].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            ax.set_title(f"Pie Chart of {column}")
            st.pyplot(fig)

    # -------------------
    # 3. Correlation & Regression
    # -------------------
    elif menu == "📉 Correlation & Regression":
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) > 1:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        if len(numeric_cols) >= 2:
            st.subheader("Scatter Plot with Regression")
            x_col = st.selectbox("Select X-axis", numeric_cols, index=0)
            y_col = st.selectbox("Select Y-axis", numeric_cols, index=1)

            fig, ax = plt.subplots()
            sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)

            # Regression line
            X = df[[x_col]].dropna()
            y = df[y_col].dropna()
            if len(X) > 1 and len(y) > 1:
                model = LinearRegression()
                model.fit(X, y[:len(X)])
                y_pred = model.predict(X)
                ax.plot(X, y_pred, color="red")
                st.write(f"Regression: {y_col} = {model.coef_[0]:.2f} × {x_col} + {model.intercept_:.2f}")

            ax.set_title(f"{x_col} vs {y_col}")
            st.pyplot(fig)
   
    # -------------------
    # 4. Advanced Business Analysis
    # -------------------

       # Sales Funnel (Static Example)
        st.subheader("Sales Funnel (Example)")
        funnel_data = {"Stage": ["Visitors", "Added to Cart", "Checkout", "Purchased"],
                       "Count": [1000, 600, 300, 150]}
        funnel_df = pd.DataFrame(funnel_data)
        fig, ax = plt.subplots()
        sns.barplot(x="Count", y="Stage", data=funnel_df, ax=ax, palette="Blues_r")
        ax.set_title("Sales Funnel")
        st.pyplot(fig)

