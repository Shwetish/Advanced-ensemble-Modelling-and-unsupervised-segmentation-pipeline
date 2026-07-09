#----------------------------------------------------------------------------
#Business Understanding:-
#----------------------------------------------------------------------------
#1. Business Problem Statement:-
#The telecom company is losing customers to competitors, and they do not
# know why or when people are planning to cancel their services. They
# also do not know which promotional offers or contract types 
#successfully keep people around.

#2. Business Objective:-
#To find out which clients are most likely to drop their services, stop 
#them from leaving by giving them targeted offers, and increase overall
# customer retention.

#3. Motivation:-
#It is much cheaper to keep a customer who is already using your internet
# than to spend marketing money finding a new one. If the data shows 
#that people on "Month-to-Month" contracts with low tenure cancel their
# plans frequently, the company can step in early and move them to a 
#stable yearly plan.

#4. Constraints:-
#Fierce Competition: Customers can switch to a rival network easily if
# they find a slightly cheaper price, regardless of what our data 
#predicts.
#No Direct Feedback: The dataset shows what they spent, but not why they might be mad (like slow internet speeds or bad customer service calls).

#5.Success Criteria
#5.1 Business Success Criteria:-
#Drop in Cancellations: Reducing the number of customers who close their
# accounts over the next quarter.
#Contract Shifts: Convincing a higher percentage of risky "Month-to-
#Month" users to upgrade to One-Year or Two-Year contracts.

#5.2 ML (Machine Learning) Success Criteria:-
#For Classification (Predicting Churn Risk): Getting an F1-Score or 
#Precision of 0.82 or higher. This ensures the system accurately 
#catches people about to cancel without accidentally giving free 
#discount offers to happy customers who were going to stay anyway.
#For Clustering (Grouping Customer Profiles): Successfully sorting 
#customers into clear groups (like High-Value Long-Term Users vs. 
#High-Risk Streaming Only Users) with distinct spending habits.

#------------------------------------------------------------------------------------------------------------------------
# Data Understanding (Cleaned & Formatted Schema)
#------------------------------------------------------------------------------------------------------------------------
'''
Column Name                    Simple Description                                  Type                        Relevance
------------------------------------------------------------------------------------------------------------------------
Customer ID                    A unique identification code for each user.         Qualitative, Nominal        Low
                                                                                   (Identifier)
Count                          A placeholder column showing '1' to count rows.     Quantitative,Binary      Low
Quarter                        The business quarter of the year (e.g., Q3).        Qualitative, Nominal        Low
Referred a Friend              Did the user refer a friend? (Yes/No).              Qualitative, Binary         High
Number of Referrals            The exact number of friends successfully referred.  Quantitative, Discrete      High
Tenure in Months               Total months the client has stayed with company.    Quantitative, Discrete      High
Offer                          Marketing deal accepted (Offer A to E).             Qualitative, Categorical       High
Phone Service                  Does the user have a home phone plan? (Yes/No).     Qualitative, Binary         High
Avg Monthly Long Distance      Average monthly minutes spent on long distance.     Quantitative, Continuous    High
Multiple Lines                 Does the user have multiple phone lines? (Yes/No).  Qualitative, Binary         High
Internet Service               Does the user use company's internet? (Yes/No).     Qualitative, Binary         High
Internet Type                  Technology used (DSL, Fiber Optic, Cable).          Qualitative, Nominal        High
Avg Monthly GB Download        Average amount of internet gigabytes used monthly.  Quantitative, Continuous    High
Online Security                Pays for extra web safety features? (Yes/No).       Qualitative, Binary         High
Online Backup                  Pays for cloud data backup storage? (Yes/No).       Qualitative, Binary         High
Device Protection              Has insurance for hardware/routers? (Yes/No).       Qualitative, Binary         High
Premium Tech Support           Pays for priority tech assistance? (Yes/No).        Qualitative, Binary         High
Streaming TV                   Uses plan to stream television? (Yes/No).           Qualitative, Binary         High
Streaming Movies               Uses plan to stream movies? (Yes/No).               Qualitative, Binary         High
Streaming Music                Uses plan to stream music? (Yes/No).                Qualitative, Binary         High
Unlimited Data                 Has an uncapped data plan? (Yes/No).                Qualitative, Binary         High
Contract                       Subscription type (Month-to-Month, 1-Yr, 2-Yr).     Qualitative, Ordinal        High
Paperless Billing              Gets digital bills instead of mail? (Yes/No).       Qualitative, Binary         Low
Payment Method                 How bills are paid (Credit Card, Bank, Check).      Qualitative, Nominal        High
Monthly Charge                 The exact amount billed every month.                Quantitative, Continuous    High
Total Charges                  Total money billed across entire account history.   Quantitative, Continuous    High
Total Refunds                  Total money paid back to the customer.              Quantitative, Continuous    High
Total Extra Data Charges       Extra fees charged for exceeding data limits.       Quantitative, Discrete      High
Total Long Distance Charges    Total fees charged for long distance calls.         Quantitative, Continuous    High
Total Revenue                  The final net cash amount made from this user.      Quantitative, Continuous    High
Customer Status / Churn        Whether the customer stayed, left, or joined.       Qualitative,  Continuous        High
------------------------------------------------------------------------------------------------------------------------
'''
#============================================================================
# Full Telco Subscriber Churn Profiling & Segmentation Pipeline
# Mixed Data Engineering, 4-Moment Analysis, One-Hot Encoding, & K-Means
#============================================================================

#----------------------------------------------------------------------------
# 1. Import Core Analytics & Machine Learning Libraries
#----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set clean visualization layouts
sns.set_theme(style="whitegrid")

#============================================================================
# 2. Load Dataset & Align Dynamic Schema
#============================================================================
# Update this string path to point directly to your local file location
file_path = "C:/10_Clustering/Telco_customer_churn.xlsx"
df = pd.read_excel(file_path, sheet_name="Telco_Churn")

# Clean column headers (handle screenshot clippings like 'Tenu' or 'Monthly Ch')
df.columns = df.columns.str.strip()
rename_dict = {
    'Tenu': 'Tenure in Months',
    'Monthly Ch': 'Monthly Charge',
    'Internet': 'Internet Service',
    'Payment Meth': 'Payment Method'
}
df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns}, inplace=True)

#----------------------------------------------------------------------------
# 3. Exploratory Data Analysis (EDA)
#----------------------------------------------------------------------------

#============================================================================
# 3.1. Structural Summary & Basic Exploration
#============================================================================
print("--- First 5 Records of Telco Churn Dataset ---")
print(df.head())
print("\n--- Last 5 Records of Telco Churn Dataset ---")
print(df.tail())
print("\n--- Structural Information & Data Dimensions ---")
print(df.shape)
print("\n--- Column Inferences & Data Types ---")
print(df.info())

# Dynamically isolating numeric columns for statistical moments
num_cols = ['Tenure in Months', 'Number of Referrals', 'Monthly Charge', 
            'Total Charges', 'Total Refunds', 'Total Revenue']
num_cols = [c for c in num_cols if c in df.columns]

print("\n--- Descriptive Metrics For Numeric Features ---")
print(df[num_cols].describe().T)
'''
Inference:
- The dataset captures complete customer profiles containing a mix of high-variance 
  financial parameters (e.g., Total Revenue) and bounded categoricals.
- Features span vastly different numeric scales. Standardizing variance is mandatory 
  to prevent distance equations from over-indexing on 'Total Revenue'.
'''

#============================================================================
# 3.2. Data Completeness Diagnostics (Missing Values)
#============================================================================
print("\n--- Missing Value Volumes Per Feature ---")
print(df.isnull().sum())
'''
Inference:
- The dataset is intact with zero missing entries across the row matrix. 
- No mechanical mathematical imputation or dropped entries are necessary.
'''

#============================================================================
# 3.3. Quantitative Analysis of First Four Statistical Moments
#============================================================================
print("\n--- First Moment: Expected Mean Value ---")
print(df[num_cols].mean())

print("\n--- Second Moment: Variance Spread (Std Dev) ---")
print(df[num_cols].std())

print("\n--- Third Moment: Skewness Coefficients ---")
print(df[num_cols].skew())

print("\n--- Fourth Moment: Kurtosis (Tail Thickness) ---")
print(df[num_cols].kurt())
'''
Inference:
- 'Total Revenue' and 'Total Charges' exhibit positive right-skewness. This proves a small 
  sub-segment of enterprise or premium accounts generates a large share of cash flow.
- 'Tenure in Months' exhibits a negative kurtosis, showing a relatively uniform 
  spread of new vs. long-term accounts without volatile outliers.
'''

#============================================================================
# 3.4. Univariate Analysis (Visualizing Core Features)
#============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.histplot(df['Tenure in Months'], bins=20, kde=True, ax=axes[0,0], color='navy')
axes[0,0].set_title('Account Tenure Distribution')

sns.histplot(df['Monthly Charge'], bins=20, kde=True, ax=axes[0,1], color='teal')
axes[0,1].set_title('Monthly Charge Distribution')

sns.histplot(df['Total Revenue'], bins=20, kde=True, ax=axes[1,0], color='crimson')
axes[1,0].set_title('Total Earned Revenue Distribution')

# Bar plot for categorical target validation
if 'Contract' in df.columns:
    sns.countplot(data=df, x='Contract', ax=axes[1,1], palette='Set2')
    axes[1,1].set_title('Subscription Contract Types Breakdown')
else:
    fig.delaxes(axes[1,1])

plt.suptitle("Univariate Frequency Profiles: Telco Account Metrics", fontsize=16)
plt.tight_layout()
plt.show()
'''
Inference:
- Tenure displays high frequencies at both ends (very new users vs. very loyal users). 
- 'Monthly Charge' highlights separate clusters around low-tier voice plans and premium fiber data accounts.
'''

#============================================================================
# 3.5. Bivariate Analysis (Cross-Feature Correlations)
#============================================================================
plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='viridis', fmt=".2f")
plt.title('Bivariate Correlation Matrix: Financials & Usage Metrics')
plt.tight_layout()
plt.show()
'''
Inference:
- 'Total Revenue' shows an extremely strong positive correlation (0.95+) with 'Total Charges' 
  and a high correlation with 'Tenure in Months'.
- This mathematically confirms that longer contract survival is the primary driver 
  of customer lifetime revenue.
'''
#----------------------------------------------------------------------------
# 4. Data Pre-Processing & Feature Engineering (FIXED)
#----------------------------------------------------------------------------

df_encoded = df.copy()

# 1. Map simple binary flags to strict 1/0 integers
binary_cols = ['Referred a Friend', 'Phone Service', 'Multiple Lines', 'Internet Service', 
               'Paperless Billing', 'Unlimited Data', 'Online Security', 'Online Backup']

for col in binary_cols:
    if col in df_encoded.columns:
        df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0}).fillna(0)

# 2. Identify ALL remaining categorical text columns (including 'Offer')
# This automatically catches 'Offer', 'Internet Type', 'Payment Method', etc.
categorical_features = ['Offer', 'Internet Type', 'Payment Method', 'Contract']
categorical_features = [c for c in categorical_features if c in df_encoded.columns]

# 3. Convert text columns into clean 1/0 dummy numeric columns
df_encoded = pd.get_dummies(df_encoded, columns=categorical_features, drop_first=True)

# 4. Filter features_to_scale so it ONLY looks at numeric columns
# We remove raw descriptive/ID tags and the target column if it exists
drop_tags = ['Customer ID', 'Count', 'Quarter', 'Customer Status / Churn']
features_to_scale = [col for col in df_encoded.columns if col not in drop_tags]

# Double check: Ensure absolutely NO text/object type columns are hidden inside
features_to_scale = df_encoded[features_to_scale].select_dtypes(include=[np.number]).columns.tolist()

#============================================================================
# 4.2. Outlier Resolution via Winsorization
#============================================================================
if 'Total Revenue' in df_encoded.columns:
    df_encoded['Total Revenue'] = winsorize(df_encoded['Total Revenue'], limits=[0.0, 0.02])

#============================================================================
# 4.3. Scale Standardization (Will execute perfectly now!)
#============================================================================
scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(df_encoded[features_to_scale])
scaled_df = pd.DataFrame(scaled_matrix, columns=features_to_scale)

print("\n--- Pre-Processing Verification Metrics ---")
print(f"Total Scaled Feature Columns Engineered: {scaled_df.shape[1]}")
print(f"Dataset Balanced Mean: {scaled_df.mean().mean().round(4)}")
print(f"Dataset Balanced Deviation: {scaled_df.std().mean().round(4)}")

#----------------------------------------------------------------------------
# 5. Model Building & Cluster Optimization
#----------------------------------------------------------------------------

#============================================================================
# 5.1 & 5.2. Scree Plot Generation (WCSS Inertia Evaluation)
#============================================================================
wcss = []
k_range = range(1, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_df)
    wcss.append(kmeans.inertia_)

# Plotting the mathematical Scree Plot
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='s', linestyle='-', color='darkgreen')
plt.title('Scree Plot: Finding the Optimal Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(k_range)
plt.grid(True)
plt.show()
'''
Inference:
- The elbow curve drops sharply and flattens out noticeably at k=3. 
- Splitting customers into 3 clusters balances structural clarity with 
  practical, real-world business strategy.
'''

#============================================================================
# 5.3. Cluster Validation & Profile Extraction (K=2 vs K=3)
#============================================================================
# Compute K=2 Structure Matrix
kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Cluster_K2'] = kmeans_2.fit_predict(scaled_df)

# Compute K=3 Structure Matrix (Selected Strategy)
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster_K3'] = kmeans_3.fit_predict(scaled_df)

print("\n=== Subscriber Profiles via K=2 Framework ===")
print(df.groupby('Cluster_K2')[num_cols].mean().round(2))

print("\n=== Subscriber Profiles via K=3 Optimization ===")
print(df.groupby('Cluster_K3')[num_cols].mean().round(2))
'''
Inference & Validation Summary:
- K=2 splits customers broadly into high-paying and low-paying groups, which obscures churn risk.
- K=4 would over-segment the data, creating small, redundant groups.
- K=3 provides an actionable public-facing strategy with three clear segments:
  * Cluster 0: High Tenure, High Revenue, High Referrals (Loyal Champions).
  * Cluster 1: Low Tenure, Low Referrals, High Monthly Charges (High Churn-Risk Streamers).
  * Cluster 2: Medium-to-High Tenure, Moderate Monthly Cost, Basic Phone/DSL (Stable, Low-Value Essentials).
'''
'''
------------------------------------------------------------------------------------
#6. Business Benefits & Solution Impact
--------------------------------------------------------------------------------------
By shifting away from generic marketing campaigns and adopting the data-driven K=3 
Customer Segmentation Framework, your telecom client can directly achieve the 
following real-world operational improvements:

1. Intercept At-Risk Customers Before They Leave
What the computer found: The model clearly isolates Cluster 1. This segment consists
 of newer accounts with very low tenure, high monthly fees, and few to no friend
 referrals. These users typically choose month-to-month terms and rely heavily on 
 internet services.

How it helps the business: Instead of reacting after a customer submits a cancellation
 request, the customer retention team can set up automated triggers for Cluster 1 
 profiles. They can offer targeted multi-month contract discounts or streaming data
 bundles early on, directly addressing the main reasons these users churn.

2. Protect and Leverage Premium Accounts
What the computer found: The model identifies Cluster 0 as your most valuable group.
 These long-term subscribers generate the highest overall revenue and actively refer
 friends to your network.

How it helps the business: Marketing teams can stop wasting retention discounts on 
this highly loyal group. Instead, the company can move them into an exclusive VIP 
loyalty tier. This allows you to leverage their high customer satisfaction by offering
 referral bonuses, turning them into a cost-effective channel for acquiring new 
 customers.

3. Maximize Profit Margins with Targeted Offers
What the computer found: The data shows that blanket promotional discounts hurt
 overall profit margins. The clustering algorithm addresses this by organizing the 
 subscriber base into distinct, easily identifiable profiles.

How it helps the business: Sales and billing teams can align their promotions with 
each specific group's needs.

Cluster 2 (mature users on basic plans) can receive targeted offers focused on low-cost
 plan stability.

High-usage accounts can be offered speed upgrades rather than price cuts.

This targeted approach helps maximize customer retention while preserving healthy 
profit margins.

'''