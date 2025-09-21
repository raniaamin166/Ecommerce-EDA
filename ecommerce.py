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
        "🛍️ E-commerce Insights",
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
    # 4. E-commerce Insights
    # -------------------
    elif menu == "🛍️ E-commerce Insights":
        if "OrderDate" in df.columns and "Sales" in df.columns:
            df["OrderDate"] = pd.to_datetime(df["OrderDate"])

            # Sales Over Time
            st.subheader("Sales Over Time")
            sales_time = df.groupby(df["OrderDate"].dt.to_period("M"))["Sales"].sum()
            sales_time.index = sales_time.index.to_timestamp()  # ✅ Fix for plotting
            fig, ax = plt.subplots()
            sales_time.plot(ax=ax, marker="o", color="teal")
            ax.set_title("Monthly Sales Trend")
            st.pyplot(fig)

            # Sales Heatmap (Day vs Hour)
            st.subheader("Sales Heatmap (Day vs Hour)")
            df["Day"] = df["OrderDate"].dt.day_name()
            df["Hour"] = df["OrderDate"].dt.hour
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            pivot = df.pivot_table(index="Day", columns="Hour", values="Sales", aggfunc="sum").reindex(day_order)
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(pivot, cmap="YlGnBu", ax=ax)
            ax.set_title("Sales Heatmap")
            st.pyplot(fig)

        if "Product" in df.columns and "Sales" in df.columns:
            # Top Products
            st.subheader("Top 10 Products by Sales")
            top_products = df.groupby("Product")["Sales"].sum().nlargest(10)
            fig, ax = plt.subplots()
            top_products.plot(kind="barh", ax=ax, color="purple")
            ax.set_title("Top Products by Sales")
            st.pyplot(fig)

        if "Category" in df.columns and "Sales" in df.columns:
            # Sales by Category
            st.subheader("Sales by Category")
            fig, ax = plt.subplots()
            sns.boxplot(x="Category", y="Sales", data=df, ax=ax, palette="Set2")
            ax.set_title("Sales by Category")
            st.pyplot(fig)

        if "Region" in df.columns:
            # Orders by Region
            st.subheader("Orders by Region")
            fig, ax = plt.subplots()
            df["Region"].value_counts().plot(kind="bar", ax=ax, color="orange")
            ax.set_title("Orders by Region")
            st.pyplot(fig)

    # -------------------
    # 5. Advanced Business Analysis
    # -------------------
   # 4. Advanced Business Analysis
# -----------------------------
elif menu == "📊 Advanced Business Analysis":
    if "OrderDate" in df.columns and "Sales" in df.columns:
        df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")

        # Sales Over Time (Fixed)
        st.subheader("📈 Sales Over Time")
        sales_time = df.groupby(df["OrderDate"].dt.to_period("M"))["Sales"].sum()
        sales_time.index = sales_time.index.astype(str)  # convert PeriodIndex to string
        fig, ax = plt.subplots()
        sales_time.plot(ax=ax, marker="o", color="teal")
        ax.set_title("Monthly Sales Trend")
        ax.set_xlabel("Month")
        ax.set_ylabel("Total Sales")
        st.pyplot(fig)

        # Sales Heatmap (Day vs Hour)
        st.subheader("🔥 Sales Heatmap (Day vs Hour)")
        df["Day"] = df["OrderDate"].dt.day_name()
        df["Hour"] = df["OrderDate"].dt.hour
        pivot = df.pivot_table(index="Day", columns="Hour", values="Sales", aggfunc="sum")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(pivot, cmap="YlGnBu", ax=ax)
        ax.set_title("Sales Heatmap")
        st.pyplot(fig)

    if "Product" in df.columns and "Sales" in df.columns:
        # Top Products
        st.subheader("🏆 Top 10 Products by Sales")
        top_products = df.groupby("Product")["Sales"].sum().nlargest(10)
        fig, ax = plt.subplots()
        top_products.plot(kind="barh", ax=ax, color="purple")
        ax.set_title("Top Products by Sales")
        st.pyplot(fig)

    if "Category" in df.columns and "Sales" in df.columns:
        # Sales by Category
        st.subheader("📦 Sales by Category")
        fig, ax = plt.subplots()
        sns.boxplot(x="Category", y="Sales", data=df, ax=ax, palette="Set2")
        ax.set_title("Sales by Category")
        st.pyplot(fig)

    if "Region" in df.columns:
        # Orders by Region
        st.subheader("🌍 Orders by Region")
        fig, ax = plt.subplots()
        df["Region"].value_counts().plot(kind="bar", ax=ax, color="orange")
        ax.set_title("Orders by Region")
        st.pyplot(fig)


            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(cohort_pivot, annot=True, fmt=".0%", cmap="YlGnBu", ax=ax)
            ax.set_title("Cohort Analysis")
            st.pyplot(fig)
