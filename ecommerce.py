import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# App Title
st.title("E-commerce Data Analysis Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your E-commerce CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ------------------------------
    # 📌 Dataset Overview
    # ------------------------------
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe(include="all"))

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # ------------------------------
    # 📊 Column-wise Analysis
    # ------------------------------
    st.subheader("Column-wise Analysis")
    column = st.selectbox("Select a column for analysis", df.columns)

    if pd.api.types.is_numeric_dtype(df[column]):
        # Histogram
        st.write(f"Histogram of **{column}**")
        fig, ax = plt.subplots()
        sns.histplot(df[column].dropna(), bins=20, kde=True, color="skyblue", ax=ax)
        st.pyplot(fig)

        # Boxplot
        st.write(f"Boxplot of **{column}**")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[column], color="lightgreen", ax=ax)
        st.pyplot(fig)

    else:
        # Bar Chart
        st.write(f"Bar Chart of **{column}**")
        fig, ax = plt.subplots()
        df[column].value_counts().head(10).plot(kind="bar", ax=ax, color="coral")
        st.pyplot(fig)

        # Pie Chart
        st.write(f"Pie Chart of **{column}**")
        fig, ax = plt.subplots()
        df[column].value_counts().head(5).plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

    # ------------------------------
    # 🔥 Correlation & Scatter Analysis
    # ------------------------------
    st.subheader("Correlation Heatmap (Numeric Columns)")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.subheader("Scatter Plot with Regression Line")
    if len(numeric_cols) >= 2:
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

        st.pyplot(fig)

    # ------------------------------
    # 📦 E-commerce Insights
    # ------------------------------
    st.subheader("E-commerce Insights")

    if {"Product", "Quantity", "UnitPrice"}.issubset(df.columns):
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

        # Top Products by Revenue
        st.write("Top 10 Products by Revenue")
        top_products = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top_products)

        # Revenue by Category
        if "Category" in df.columns:
            st.write("Revenue by Category")
            revenue_by_category = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
            st.bar_chart(revenue_by_category)

        # Sales Trend Over Time
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            sales_trend = df.groupby(df["Date"].dt.to_period("M"))["Revenue"].sum()
            sales_trend.index = sales_trend.index.astype(str)  # convert PeriodIndex to str
            st.write("Monthly Sales Trend")
            st.line_chart(sales_trend)

