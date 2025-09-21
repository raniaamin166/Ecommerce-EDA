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
            fig, ax = plt.subplots()
            sales_time.plot(ax=ax, marker="o", color="teal")
            ax.set_title("Monthly Sales Trend")
            st.pyplot(fig)

            # Sales Heatmap (Day vs Hour)
            st.subheader("Sales Heatmap (Day vs Hour)")
            df["Day"] = df["OrderDate"].dt.day_name()
            df["Hour"] = df["OrderDate"].dt.hour
            pivot = df.pivot_table(index="Day", columns="Hour", values="Sales", aggfunc="sum")
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
    elif menu == "🔥 Advanced Business Analysis":
        if "CustomerID" in df.columns and "Sales" in df.columns:
            # CLV
            st.subheader("Customer Lifetime Value Distribution")
            clv = df.groupby("CustomerID")["Sales"].sum()
            fig, ax = plt.subplots()
            sns.histplot(clv, kde=True, bins=30, ax=ax, color="darkcyan")
            ax.set_title("CLV Distribution")
            st.pyplot(fig)

        if "Product" in df.columns and "Sales" in df.columns:
            # Pareto
            st.subheader("Pareto Analysis (Products)")
            sales_by_product = df.groupby("Product")["Sales"].sum().sort_values(ascending=False)
            cum_percent = sales_by_product.cumsum() / sales_by_product.sum() * 100
            fig, ax1 = plt.subplots()
            sales_by_product.plot(kind="bar", ax=ax1, color="lightblue")
            ax2 = ax1.twinx()
            cum_percent.plot(ax=ax2, color="red", marker="D")
            ax1.set_ylabel("Sales")
            ax2.set_ylabel("Cumulative %")
            ax1.set_title("Pareto Chart")
            st.pyplot(fig)

        if "CustomerID" in df.columns and "OrderDate" in df.columns and "Sales" in df.columns:
            # RFM
            st.subheader("RFM Segmentation")
            df["OrderDate"] = pd.to_datetime(df["OrderDate"])
            ref_date = df["OrderDate"].max() + pd.Timedelta(days=1)
            rfm = df.groupby("CustomerID").agg({
                "OrderDate": lambda x: (ref_date - x.max()).days,
                "CustomerID": "count",
                "Sales": "sum"
            })
            rfm.columns = ["Recency", "Frequency", "Monetary"]
            fig, ax = plt.subplots()
            sns.scatterplot(data=rfm, x="Recency", y="Frequency", size="Monetary", ax=ax, alpha=0.6)
            ax.set_title("RFM Segmentation")
            st.pyplot(fig)

        # Sales Funnel (Static Example)
        st.subheader("Sales Funnel (Example)")
        funnel_data = {"Stage": ["Visitors", "Added to Cart", "Checkout", "Purchased"],
                       "Count": [1000, 600, 300, 150]}
        funnel_df = pd.DataFrame(funnel_data)
        fig, ax = plt.subplots()
        sns.barplot(x="Count", y="Stage", data=funnel_df, ax=ax, palette="Blues_r")
        ax.set_title("Sales Funnel")
        st.pyplot(fig)

        if "CustomerID" in df.columns and "OrderDate" in df.columns:
            # Cohort
            st.subheader("Cohort Retention Heatmap")
            df["OrderMonth"] = df["OrderDate"].dt.to_period("M")
            df["CohortMonth"] = df.groupby("CustomerID")["OrderMonth"].transform("min")
            cohort = df.groupby(["CohortMonth", "OrderMonth"])["CustomerID"].nunique().reset_index()
            cohort_pivot = cohort.pivot_table(index="CohortMonth", columns="OrderMonth", values="CustomerID")
            cohort_pivot = cohort_pivot.divide(cohort_pivot.iloc[:, 0], axis=0)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(cohort_pivot, annot=True, fmt=".0%", cmap="YlGnBu", ax=ax)
            ax.set_title("Cohort Analysis")
            st.pyplot(fig)
