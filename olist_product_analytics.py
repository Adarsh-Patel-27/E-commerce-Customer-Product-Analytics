# ============================================================
# OLIST E-COMMERCE CUSTOMER & PRODUCT ANALYTICS PROJECT
# ============================================================
# Project workflow:
# Dataset -> Cleaning -> EDA -> Customer Analysis -> RFM
# -> Product Analysis -> Cohort/Retention -> Reviews
# -> Visualizations -> Exported Results
#
# How to use:
# 1. Put this .py file in the same folder as the Olist CSV files.
# 2. Run: python olist_product_analytics.py
# 3. The script automatically creates an "olist_analysis_output"
#    folder containing CSV results and PNG graphs.
#
# Required libraries:
# pandas, numpy, matplotlib, seaborn
#
# Install if needed:
# pip install pandas numpy matplotlib seaborn
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# -----------------------------
# 1. SETTINGS
# -----------------------------

# The script expects the CSV files in the same folder as this script.
BASE_DIR = Path(__file__).resolve().parent

# If you are running in Google Colab/Jupyter and __file__ does not exist,
# replace BASE_DIR above with:
# BASE_DIR = Path("/content/olist_data")

OUTPUT_DIR = BASE_DIR / "olist_analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 120

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


# -----------------------------
# 2. HELPER FUNCTIONS
# -----------------------------

def save_plot(filename):
    """Save the current matplotlib figure and close it."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close()


def load_csv(filename):
    """Load an Olist CSV file and stop with a useful error if missing."""
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"\nCould not find: {filename}\n"
            f"Expected it at: {path}\n"
            f"Put all Olist CSV files in the same folder as this script."
        )
    return pd.read_csv(path)


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# -----------------------------
# 3. LOAD DATA
# -----------------------------

print_section("LOADING DATA")

customers = load_csv("olist_customers_dataset.csv")
orders = load_csv("olist_orders_dataset.csv")
order_items = load_csv("olist_order_items_dataset.csv")
payments = load_csv("olist_order_payments_dataset.csv")
reviews = load_csv("olist_order_reviews_dataset.csv")
products = load_csv("olist_products_dataset.csv")
sellers = load_csv("olist_sellers_dataset.csv")
category_translation = load_csv("product_category_name_translation.csv")

datasets = {
    "customers": customers,
    "orders": orders,
    "order_items": order_items,
    "payments": payments,
    "reviews": reviews,
    "products": products,
    "sellers": sellers,
    "category_translation": category_translation
}

for name, df in datasets.items():
    print(f"{name:25s}: {df.shape[0]:>8,} rows x {df.shape[1]:>3} columns")


# -----------------------------
# 4. DATA QUALITY CHECK
# -----------------------------

print_section("DATA QUALITY CHECK")

quality_rows = []

for name, df in datasets.items():
    quality_rows.append({
        "dataset": name,
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum())
    })

quality_report = pd.DataFrame(quality_rows)
print(quality_report)

quality_report.to_csv(
    OUTPUT_DIR / "data_quality_report.csv",
    index=False
)


# -----------------------------
# 5. DATE CONVERSION
# -----------------------------

print_section("DATE CONVERSION")

date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for col in date_columns:
    orders[col] = pd.to_datetime(
        orders[col],
        errors="coerce"
    )

print(orders[date_columns].dtypes)


# -----------------------------
# 6. BASIC ORDER STATUS CHECK
# -----------------------------

print_section("ORDER STATUS")

print(orders["order_status"].value_counts())

order_status = (
    orders["order_status"]
    .value_counts()
    .reset_index()
)

order_status.columns = ["order_status", "order_count"]

order_status.to_csv(
    OUTPUT_DIR / "order_status_summary.csv",
    index=False
)

plt.figure()
sns.barplot(
    data=order_status,
    x="order_status",
    y="order_count"
)
plt.title("Orders by Order Status")
plt.xlabel("Order Status")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
save_plot("01_order_status.png")


# -----------------------------
# 7. BUILD MAIN ANALYSIS TABLE
# -----------------------------

print_section("BUILDING MAIN ANALYSIS DATASET")

# Orders + customer information
analysis = orders.merge(
    customers,
    on="customer_id",
    how="left"
)

# Add order-item information
analysis = analysis.merge(
    order_items,
    on="order_id",
    how="left"
)

# Add product information
analysis = analysis.merge(
    products,
    on="product_id",
    how="left"
)

# Add English category names
analysis = analysis.merge(
    category_translation,
    on="product_category_name",
    how="left"
)

# Item-level total value
analysis["item_total_value"] = (
    analysis["price"].fillna(0)
    + analysis["freight_value"].fillna(0)
)

print("Main analysis dataset:", analysis.shape)


# -----------------------------
# 8. DELIVERED ORDER DATA
# -----------------------------

delivered = analysis[
    analysis["order_status"] == "delivered"
].copy()

print("Delivered analysis rows:", len(delivered))


# -----------------------------
# 9. CORE BUSINESS KPIs
# -----------------------------

print_section("CORE BUSINESS KPIs")

total_orders = delivered["order_id"].nunique()
total_customers = delivered["customer_unique_id"].nunique()
total_revenue = delivered["item_total_value"].sum()

order_values = (
    delivered
    .groupby("order_id")["item_total_value"]
    .sum()
)

average_order_value = order_values.mean()

customer_order_counts = (
    delivered
    .groupby("customer_unique_id")["order_id"]
    .nunique()
)

repeat_customer_rate = (
    (customer_order_counts > 1).mean() * 100
)

kpis = pd.DataFrame({
    "Metric": [
        "Delivered Orders",
        "Unique Customers",
        "Total Revenue",
        "Average Order Value",
        "Repeat Customer Rate (%)"
    ],
    "Value": [
        total_orders,
        total_customers,
        total_revenue,
        average_order_value,
        repeat_customer_rate
    ]
})

print(kpis)

kpis.to_csv(
    OUTPUT_DIR / "core_kpis.csv",
    index=False
)


# -----------------------------
# 10. MONTHLY REVENUE
# -----------------------------

print_section("MONTHLY REVENUE")

delivered["order_month"] = (
    delivered["order_purchase_timestamp"]
    .dt.to_period("M")
)

monthly_revenue = (
    delivered
    .groupby("order_month")["item_total_value"]
    .sum()
    .reset_index()
)

monthly_revenue["order_month"] = (
    monthly_revenue["order_month"].astype(str)
)

print(monthly_revenue)

monthly_revenue.to_csv(
    OUTPUT_DIR / "monthly_revenue.csv",
    index=False
)

plt.figure()
sns.lineplot(
    data=monthly_revenue,
    x="order_month",
    y="item_total_value",
    marker="o"
)
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
save_plot("02_monthly_revenue.png")


# -----------------------------
# 11. MONTHLY ORDERS
# -----------------------------

monthly_orders = (
    delivered
    .groupby("order_month")["order_id"]
    .nunique()
    .reset_index()
)

monthly_orders["order_month"] = (
    monthly_orders["order_month"].astype(str)
)

monthly_orders.to_csv(
    OUTPUT_DIR / "monthly_orders.csv",
    index=False
)

plt.figure()
sns.lineplot(
    data=monthly_orders,
    x="order_month",
    y="order_id",
    marker="o"
)
plt.title("Monthly Order Trend")
plt.xlabel("Month")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
save_plot("03_monthly_orders.png")


# -----------------------------
# 12. CUSTOMER-LEVEL SUMMARY
# -----------------------------

print_section("CUSTOMER ANALYSIS")

customer_summary = (
    delivered
    .groupby("customer_unique_id")
    .agg(
        total_orders=("order_id", "nunique"),
        total_spent=("item_total_value", "sum"),
        average_order_value=("item_total_value", "mean"),
        first_order_date=(
            "order_purchase_timestamp",
            "min"
        ),
        last_order_date=(
            "order_purchase_timestamp",
            "max"
        )
    )
    .reset_index()
)

customer_summary["customer_type"] = np.where(
    customer_summary["total_orders"] > 1,
    "Repeat",
    "One-time"
)

print(customer_summary.head())

customer_summary.to_csv(
    OUTPUT_DIR / "customer_summary.csv",
    index=False
)


# -----------------------------
# 13. ONE-TIME VS REPEAT
# -----------------------------

customer_type_counts = (
    customer_summary["customer_type"]
    .value_counts()
    .reset_index()
)

customer_type_counts.columns = [
    "customer_type",
    "customer_count"
]

print(customer_type_counts)

customer_type_counts.to_csv(
    OUTPUT_DIR / "one_time_vs_repeat_customers.csv",
    index=False
)

plt.figure()
sns.barplot(
    data=customer_type_counts,
    x="customer_type",
    y="customer_count"
)
plt.title("One-Time vs Repeat Customers")
plt.xlabel("Customer Type")
plt.ylabel("Number of Customers")
save_plot("04_one_time_vs_repeat.png")


# -----------------------------
# 14. PURCHASE FREQUENCY
# -----------------------------

plt.figure()
sns.histplot(
    customer_summary["total_orders"],
    bins=20,
    kde=True
)
plt.title("Customer Purchase Frequency")
plt.xlabel("Number of Orders")
plt.ylabel("Number of Customers")
save_plot("05_purchase_frequency.png")


# -----------------------------
# 15. CUSTOMER SPENDING
# -----------------------------

plt.figure()
sns.histplot(
    customer_summary["total_spent"],
    bins=50,
    kde=True
)
plt.title("Customer Spending Distribution")
plt.xlabel("Total Customer Spending")
plt.ylabel("Number of Customers")
save_plot("06_customer_spending_distribution.png")


# -----------------------------
# 16. RFM ANALYSIS
# -----------------------------

print_section("RFM CUSTOMER SEGMENTATION")

reference_date = (
    delivered["order_purchase_timestamp"].max()
    + pd.Timedelta(days=1)
)

rfm = (
    delivered
    .groupby("customer_unique_id")
    .agg(
        Recency=(
            "order_purchase_timestamp",
            lambda x: (reference_date - x.max()).days
        ),
        Frequency=("order_id", "nunique"),
        Monetary=("item_total_value", "sum")
    )
    .reset_index()
)

# Rank-based qcut is used for Frequency because many customers
# have the same number of orders.
rfm["R_score"] = pd.qcut(
    rfm["Recency"],
    5,
    labels=[5, 4, 3, 2, 1]
)

rfm["F_score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
)

rfm["M_score"] = pd.qcut(
    rfm["Monetary"],
    5,
    labels=[1, 2, 3, 4, 5]
)

rfm["R_score"] = rfm["R_score"].astype(int)
rfm["F_score"] = rfm["F_score"].astype(int)
rfm["M_score"] = rfm["M_score"].astype(int)

rfm["RFM_score"] = (
    rfm["R_score"]
    + rfm["F_score"]
    + rfm["M_score"]
)


def assign_segment(score):
    if score >= 13:
        return "High Value"
    elif score >= 10:
        return "Loyal"
    elif score >= 7:
        return "Potential"
    else:
        return "At Risk"


rfm["segment"] = rfm["RFM_score"].apply(assign_segment)

print(rfm["segment"].value_counts())

rfm.to_csv(
    OUTPUT_DIR / "customer_rfm_segments.csv",
    index=False
)


# -----------------------------
# 17. RFM SEGMENT SIZE
# -----------------------------

segment_counts = (
    rfm["segment"]
    .value_counts()
    .reset_index()
)

segment_counts.columns = [
    "segment",
    "customer_count"
]

segment_counts.to_csv(
    OUTPUT_DIR / "rfm_segment_counts.csv",
    index=False
)

plt.figure()
sns.barplot(
    data=segment_counts,
    x="segment",
    y="customer_count",
    order=segment_counts["segment"]
)
plt.title("Customer Segmentation using RFM")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)
save_plot("07_rfm_customer_segments.png")


# -----------------------------
# 18. REVENUE BY RFM SEGMENT
# -----------------------------

segment_revenue = (
    rfm
    .groupby("segment")["Monetary"]
    .agg(["sum", "mean"])
    .reset_index()
)

segment_revenue.columns = [
    "segment",
    "total_revenue",
    "average_revenue_per_customer"
]

segment_revenue["revenue_share_pct"] = (
    segment_revenue["total_revenue"]
    / segment_revenue["total_revenue"].sum()
    * 100
)

segment_revenue = segment_revenue.sort_values(
    "total_revenue",
    ascending=False
)

print(segment_revenue)

segment_revenue.to_csv(
    OUTPUT_DIR / "revenue_by_rfm_segment.csv",
    index=False
)

plt.figure()
sns.barplot(
    data=segment_revenue,
    x="segment",
    y="total_revenue"
)
plt.title("Revenue Contribution by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Revenue")
plt.xticks(rotation=20)
save_plot("08_revenue_by_rfm_segment.png")


# -----------------------------
# 19. PRODUCT CATEGORY ANALYSIS
# -----------------------------

print_section("PRODUCT CATEGORY ANALYSIS")

# Use English category name where available;
# otherwise use the original Portuguese category.
delivered["category"] = (
    delivered["product_category_name_english"]
    .fillna(delivered["product_category_name"])
    .fillna("Unknown")
)

category_analysis = (
    delivered
    .groupby("category")
    .agg(
        revenue=("item_total_value", "sum"),
        orders=("order_id", "nunique"),
        items_sold=("order_item_id", "count"),
        average_item_value=("item_total_value", "mean")
    )
    .reset_index()
)

category_analysis = category_analysis.sort_values(
    "revenue",
    ascending=False
)

category_analysis.to_csv(
    OUTPUT_DIR / "category_analysis.csv",
    index=False
)

top_categories = category_analysis.head(15).sort_values(
    "revenue"
)

plt.figure(figsize=(10, 7))
sns.barplot(
    data=top_categories,
    x="revenue",
    y="category"
)
plt.title("Top 15 Product Categories by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Product Category")
save_plot("09_top_categories_revenue.png")


# -----------------------------
# 20. CATEGORY BY ORDER VOLUME
# -----------------------------

top_order_categories = (
    category_analysis
    .sort_values("orders", ascending=False)
    .head(15)
    .sort_values("orders")
)

plt.figure(figsize=(10, 7))
sns.barplot(
    data=top_order_categories,
    x="orders",
    y="category"
)
plt.title("Top 15 Product Categories by Order Volume")
plt.xlabel("Number of Orders")
plt.ylabel("Product Category")
save_plot("10_top_categories_orders.png")


# -----------------------------
# 21. AOV BY CATEGORY
# -----------------------------

top_aov_categories = (
    category_analysis[
        category_analysis["orders"] >= 50
    ]
    .sort_values("average_item_value", ascending=False)
    .head(15)
    .sort_values("average_item_value")
)

plt.figure(figsize=(10, 7))
sns.barplot(
    data=top_aov_categories,
    x="average_item_value",
    y="category"
)
plt.title("Top Categories by Average Item Value")
plt.xlabel("Average Item Value")
plt.ylabel("Product Category")
save_plot("11_category_average_value.png")


# -----------------------------
# 22. COHORT / RETENTION ANALYSIS
# -----------------------------

print_section("COHORT AND RETENTION ANALYSIS")

customer_first_purchase = (
    delivered
    .groupby("customer_unique_id")[
        "order_purchase_timestamp"
    ]
    .min()
    .reset_index()
)

customer_first_purchase.columns = [
    "customer_unique_id",
    "first_purchase_date"
]

cohort_data = delivered.merge(
    customer_first_purchase,
    on="customer_unique_id",
    how="left"
)

cohort_data["cohort_month"] = (
    cohort_data["first_purchase_date"]
    .dt.to_period("M")
)

cohort_data["order_month"] = (
    cohort_data["order_purchase_timestamp"]
    .dt.to_period("M")
)

cohort_data["cohort_index"] = (
    (cohort_data["order_month"].dt.year -
     cohort_data["cohort_month"].dt.year) * 12
    +
    (cohort_data["order_month"].dt.month -
     cohort_data["cohort_month"].dt.month)
)

cohort_counts = (
    cohort_data
    .groupby(
        ["cohort_month", "cohort_index"]
    )["customer_unique_id"]
    .nunique()
    .reset_index()
)

cohort_table = cohort_counts.pivot(
    index="cohort_month",
    columns="cohort_index",
    values="customer_unique_id"
)

cohort_sizes = cohort_table.iloc[:, 0]

retention_table = (
    cohort_table
    .divide(cohort_sizes, axis=0)
    * 100
)

retention_table.to_csv(
    OUTPUT_DIR / "retention_cohort_table.csv"
)

plt.figure(figsize=(14, 8))
sns.heatmap(
    retention_table,
    annot=True,
    fmt=".1f",
    cmap="Blues"
)
plt.title("Customer Retention by Cohort")
plt.xlabel("Months Since First Purchase")
plt.ylabel("Customer Cohort")
save_plot("12_cohort_retention_heatmap.png")


# -----------------------------
# 23. CUSTOMER EXPERIENCE
# -----------------------------

print_section("CUSTOMER EXPERIENCE ANALYSIS")

order_review = orders[
    [
        "order_id",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
].merge(
    reviews[
        ["order_id", "review_score"]
    ],
    on="order_id",
    how="inner"
)

order_review["delivery_days"] = (
    order_review["order_delivered_customer_date"]
    -
    order_review["order_purchase_timestamp"]
).dt.total_seconds() / (60 * 60 * 24)

order_review["estimated_delivery_days"] = (
    order_review["order_estimated_delivery_date"]
    -
    order_review["order_purchase_timestamp"]
).dt.total_seconds() / (60 * 60 * 24)

order_review["delivery_delay_days"] = (
    order_review["delivery_days"]
    -
    order_review["estimated_delivery_days"]
)

order_review = order_review[
    order_review["delivery_days"].notna()
].copy()

# Remove clearly unusable negative delivery durations
order_review = order_review[
    order_review["delivery_days"] >= 0
]

delivery_review_summary = (
    order_review
    .groupby("review_score")["delivery_days"]
    .agg(["mean", "median", "count"])
    .reset_index()
)

print(delivery_review_summary)

delivery_review_summary.to_csv(
    OUTPUT_DIR / "delivery_time_by_review_score.csv",
    index=False
)

plt.figure()
sns.barplot(
    data=delivery_review_summary,
    x="review_score",
    y="mean"
)
plt.title("Average Delivery Time by Review Score")
plt.xlabel("Review Score")
plt.ylabel("Average Delivery Time (Days)")
save_plot("13_delivery_time_by_review.png")


# -----------------------------
# 24. REVIEW SCORE DISTRIBUTION
# -----------------------------

review_distribution = (
    reviews["review_score"]
    .value_counts()
    .sort_index()
    .reset_index()
)

review_distribution.columns = [
    "review_score",
    "review_count"
]

review_distribution.to_csv(
    OUTPUT_DIR / "review_score_distribution.csv",
    index=False
)

plt.figure()
sns.barplot(
    data=review_distribution,
    x="review_score",
    y="review_count"
)
plt.title("Customer Review Score Distribution")
plt.xlabel("Review Score")
plt.ylabel("Number of Reviews")
save_plot("14_review_score_distribution.png")


# -----------------------------
# 25. FREQUENCY VS SPENDING
# -----------------------------

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=customer_summary,
    x="total_orders",
    y="total_spent",
    alpha=0.35
)
plt.title("Purchase Frequency vs Customer Spending")
plt.xlabel("Number of Orders")
plt.ylabel("Total Customer Spending")
save_plot("15_frequency_vs_spending.png")


# -----------------------------
# 26. CUSTOMER METRIC CORRELATION
# -----------------------------

correlation_data = customer_summary[
    [
        "total_orders",
        "total_spent",
        "average_order_value"
    ]
]

plt.figure(figsize=(7, 5))
sns.heatmap(
    correlation_data.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Relationship Between Customer Metrics")
save_plot("16_customer_metric_correlation.png")


# -----------------------------
# 27. INSIGHT-READY TABLES
# -----------------------------

print_section("CREATING INSIGHT TABLES")

# Top customers
top_customers = (
    customer_summary
    .sort_values("total_spent", ascending=False)
    .head(100)
)

top_customers.to_csv(
    OUTPUT_DIR / "top_100_customers.csv",
    index=False
)

# Top categories
category_analysis.head(20).to_csv(
    OUTPUT_DIR / "top_20_categories.csv",
    index=False
)

# RFM segment summary
rfm_summary = (
    rfm
    .groupby("segment")
    .agg(
        customers=("customer_unique_id", "nunique"),
        average_recency=("Recency", "mean"),
        average_frequency=("Frequency", "mean"),
        average_monetary=("Monetary", "mean"),
        total_revenue=("Monetary", "sum")
    )
    .reset_index()
)

rfm_summary["revenue_share_pct"] = (
    rfm_summary["total_revenue"]
    / rfm_summary["total_revenue"].sum()
    * 100
)

rfm_summary.to_csv(
    OUTPUT_DIR / "rfm_summary.csv",
    index=False
)

print(rfm_summary)


# -----------------------------
# 28. FINAL SUMMARY
# -----------------------------

print_section("PROJECT SUMMARY")

print(f"Delivered orders       : {total_orders:,}")
print(f"Unique customers       : {total_customers:,}")
print(f"Total revenue          : {total_revenue:,.2f}")
print(f"Average order value    : {average_order_value:,.2f}")
print(f"Repeat customer rate   : {repeat_customer_rate:.2f}%")

print("\nTop 5 categories by revenue:")
print(
    category_analysis[
        ["category", "revenue", "orders"]
    ].head(5).to_string(index=False)
)

print("\nRFM segment summary:")
print(rfm_summary.to_string(index=False))

print("\nAnalysis completed successfully!")
print(f"All CSV results and PNG graphs are saved in:\n{OUTPUT_DIR}")
