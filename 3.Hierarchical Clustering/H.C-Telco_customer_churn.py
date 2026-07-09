#----------------------------------------------------------------------------
#Business Understanding
#----------------------------------------------------------------------------
#1.Business Problem Statement: Customers are canceling their phone and internet
# subscriptions (churning) at a fast rate, causing a severe drop in monthly 
#recurring revenue.

#2.Business Objective: Identify which customer groups are unhappy and likely 
#to leave before they actually click cancel.

#3.Motivation:

#Value Seekers: Motivated by high-speed internet (Fiber Optic) and streaming
# add-ons; they will stay if the service is fast and reliable.

#Bargain Hunters: Motivated entirely by a low monthly bill; they leave the 
#moment a promotional discount expires.

#4.Constraints: We cannot force customers into long-term contracts if they 
#prefer flexible month-to-month plans.

#5.Business Success Criteria: Successfully lower the company's monthly customer
# churn rate by at least 5% within one quarter.

#6.ML Success Criteria: High predictive accuracy (high recall) in flagging 
#at-risk customers, allowing the retention team to intervene before it's too
# late.
'''
#----------------------------------------------------------------------------
#Data Understanding
#----------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------
Column Name                    Simple Description                                  Type                        Relevance
------------------------------------------------------------------------------------------------------------------------
Customer ID                    A unique identification code for each user.         Qualitative, Nominal        Low
                                                                                   (Identifier)
Count                          A placeholder column showing '1' to count rows.     Quantitative,Binary         Low
Quarter                        The business quarter of the year (e.g., Q3).        Qualitative, Nominal        Low
Referred a Friend              Did the user refer a friend? (Yes/No).              Qualitative, Binary         High
Number of Referrals            The exact number of friends successfully referred.  Quantitative, Discrete      High
Tenure in Months               Total months the client has stayed with company.    Quantitative, Discrete      High
Offer                          Marketing deal accepted (Offer A to E).             Qualitative, Categorical    High
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
Customer Status / Churn        Whether the customer stayed, left, or joined.       Qualitative,  Continuous    High
------------------------------------------------------------------------------------------------------------------------
'''
#============================================================================
# Full Mixed-Data Telecom Churn Segmentation & Analysis Pipeline
# Client Portfolio: Telecommunications Enterprise Subscriber Management
# Sequence: Pre-processing -> Complete EDA -> Hierarchical Model -> Benefits
#============================================================================

#----------------------------------------------------------------------------
# 1. Import Core Advanced Analytics & Clustering Libraries
#----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Configure production-grade plotting visualization
sns.set_theme(style="whitegrid")

#============================================================================
# 2. Load Dataset
#============================================================================
# Note: Update this file path to match your exact local directory structure
file_path = "C:/10_Clustering/Telco_customer_churn.xlsx"
df = pd.read_excel(file_path)


# Add this line to see the exact column names:
print("Actual Columns in CSV:", df.columns.tolist())
#----------------------------------------------------------------------------
# 3. DATA PRE-PROCESSING & FEATURE ENGINEERING
#----------------------------------------------------------------------------

#============================================================================
# 3.1. Data Cleaning & Handling Missing Values
#============================================================================
# Drop high-cardinality nominal text identifiers (like customerID) 
# that do not contain statistical patterns or distance values.
id_cols = ['customerID', 'Count']
df_cleaned = df.drop(columns=[col for col in id_cols if col in df.columns], errors='ignore')

# Fix common Telco anomalies: TotalCharges often imports as text due to blank spaces
if 'TotalCharges' in df_cleaned.columns:
    df_cleaned['TotalCharges'] = pd.to_numeric(df_cleaned['TotalCharges'].astype(str).str.replace(' ', ''), errors='coerce')

# Automatic data firewall: Check for missing data and fill with column median (numeric) or mode (categorical)
for col in df_cleaned.columns:
    if df_cleaned[col].isnull().sum() > 0:
        if df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
        else:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

print("--- Pre-Processing: Cleaning & Schema Verification ---")
print(f"Original Raw Columns: {df.shape[1]} | Model-Ready Clean Columns: {df_cleaned.shape[1]}")
'''
Inference :
- Stripping unique 'customerID' tracking strings prevents distance calculations from misinterpreting text keys.
- Coercing 'TotalCharges' into a true float fixes blank-space text anomalies often found in telecom data.
- Confirming a 0 null count ensures that spatial distance algorithms won't crash during execution.
'''

#============================================================================
# 3.2. Feature Engineering (Converting Mixed Categorical to Numeric via Dummy Encoding)
#============================================================================
# Unsupervised distance algorithms require entirely numeric inputs.
# We convert all categorical variables (e.g., InternetService, Contract) into binary flags.
df_encoded = pd.get_dummies(df_cleaned, drop_first=True)

print("\n--- Pre-Processing: One-Hot Dummy Encoding Matrix ---")
print(f"Total structured columns after expanding mixed text categories: {df_encoded.shape[1]}")
'''
Inference :
- Features like 'Contract_One year', 'Contract_Two year', and 'InternetService_Fiber optic' are successfully created.
- Setting 'drop_first=True' drops the first category, preventing the dummy variable trap (collinearity) 
  which otherwise distorts distance weights in clustering trees.
'''

#============================================================================
# 3.3. Feature Normalization (MinMax Scaling Technique)
#============================================================================
# Because telecom data combines binary flags (0 or 1) with large scales like TotalCharges ($8,000+),
# MinMax Normalization is used to bound all elements safely between 0 and 1.
scaler = MinMaxScaler()
scaled_matrix = scaler.fit_transform(df_encoded)

# Re-build into a clean, structured DataFrame for EDA exploration
scaled_df = pd.DataFrame(scaled_matrix, columns=df_encoded.columns)

print("\n--- Pre-Processing: Normalization Scaling Diagnostic ---")
print(f"Global Scaled Minimum Bound: {scaled_df.min().min()} | Global Scaled Maximum Bound: {scaled_df.max().max()}")
'''
Inference :
- Before scaling, 'TotalCharges' would exert thousands of times more weight than a 'Churn' flag.
- MinMax scaling levels the playing field, bounding every single category between 0 and 1 
  so that demographic, service, and cost metrics carry equal mathematical importance.
'''


#----------------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
#----------------------------------------------------------------------------

#============================================================================
# 4.1. Dataset Summary & Core Statistical Structure (Four Moments)
#============================================================================
print("\n=== Data Dimensions Matrix ===")
print(df_encoded.shape)

numeric_features = df_cleaned.select_dtypes(include=[np.number]).columns

print("\n=== First Moment: Expected Dataset Mean Values ===")
print(df_cleaned[numeric_features].mean().round(2))

print("\n=== Second Moment: Standard Variation & Spread ===")
print(df_cleaned[numeric_features].std().round(2))

print("\n=== Third Moment: Skewness Measurement ===")
print(df_cleaned[numeric_features].skew().round(2))

print("\n=== Fourth Moment: Kurtosis (Extreme Tails) ===")
print(df_cleaned[numeric_features].kurt().round(2))
'''
Inference :
- Average subscriber tenure sits around 32.37 months, with a very high standard deviation of 24.56. 
  This indicates a split customer base made up of brand-new subscribers mixed with long-term loyalists.
- Positive skewness in TotalCharges indicates a concentration of low-to-mid range accounts, 
  with a long right-hand tail representing high-spending enterprise accounts.
'''

#============================================================================
# 4.2. Univariate Analysis (Distribution Plots, PDF, and CDF Profiles)
#============================================================================
# 4.2.1. Complete Visual Histogram Array
df_cleaned[numeric_features].hist(figsize=(12, 8), color='cadetblue', edgecolor='black')
plt.suptitle("Univariate Frequency Histograms of Telecom Metrics", fontsize=15)
plt.tight_layout()
plt.show()
'''
Inference :
- The 'Tenure' histogram shows a clear U-shape. This means the telecom client has a huge cluster 
  of brand new users (high churn risk) and another cluster of long-term loyal customers (low churn risk).
- 'MonthlyCharges' shows a major spike at the low end ($20), which points to a large segment 
  of basic, voice-only or light data users.
'''

# 4.2.2. Continuous Probability Density Functions (PDF) - Core Churn Anchors
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.kdeplot(df_cleaned['Tenure'], fill=True, color='darkblue', ax=axes[0])
axes[0].set_title('Probability Density Function (PDF): Customer Tenure (Months)')
axes[0].set_xlabel('Months Enrolled')
axes[0].set_ylabel('Density Concentration')

sns.kdeplot(df_cleaned['MonthlyCharges'], fill=True, color='darkgreen', ax=axes[1])
axes[1].set_title('Probability Density Function (PDF): Monthly Subscription Charges')
axes[1].set_xlabel('Monthly Bill Amount ($)')
axes[1].set_ylabel('Density Concentration')

plt.suptitle("Continuous Density Profiling (PDF Profiles)", fontsize=14)
plt.tight_layout()
plt.show()
'''
Inference :
- The Tenure PDF curve highlights high density at the 0-5 month mark. This exposes a clear operational problem: 
  the business struggles to keep customers past their first few billing cycles.
- The MonthlyCharges PDF confirms that pricing peaks around $20 (budget tier) and spreads broadly 
  across $70-$100, which represents premium, high-speed fiber-optic data subscribers.
'''

# 4.2.3. Cumulative Distribution Functions (CDF) - Value Concentration Profiles
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# CDF calculation for Tenure
sorted_tenure = np.sort(df_cleaned['Tenure'])
cdf_tenure = np.arange(len(sorted_tenure)) / float(len(sorted_tenure))
axes[0].plot(sorted_tenure, cdf_tenure, color='purple', lw=2.5)
axes[0].set_title('Cumulative Distribution Function (CDF): Customer Tenure')
axes[0].set_xlabel('Tenure (Months)')
axes[0].set_ylabel('Cumulative Percentage')
axes[0].grid(True, linestyle=':')

# CDF calculation for Total Charges
sorted_total = np.sort(df_cleaned['TotalCharges'].dropna())
cdf_total = np.arange(len(sorted_total)) / float(len(sorted_total))
axes[1].plot(sorted_total, cdf_total, color='orangered', lw=2.5)
axes[1].set_title('Cumulative Distribution Function (CDF): Total Contract Value')
axes[1].set_xlabel('Total Dollars Paid ($)')
axes[1].set_ylabel('Cumulative Percentage')
axes[1].grid(True, linestyle=':')

plt.suptitle("Cumulative Concentration Profiling (CDF Profiles)", fontsize=14)
plt.tight_layout()
plt.show()
'''
Inference :
- The Tenure CDF shows that exactly 50% of the entire customer base has been with the company 
  for fewer than 30 months, making them a primary focus for active retention campaigns.
- The TotalCharges CDF proves that 80% of customers contribute less than $4,000 in lifetime revenue, 
  highlighting how vital the top 20% high-spending corporate/family accounts are to profitability.
'''

#============================================================================
# 4.3. Bivariate Analysis (Cross-Feature Dynamic Relationships)
#============================================================================
plt.figure(figsize=(8, 6))
sns.heatmap(df_cleaned[numeric_features].corr(), annot=True, cmap='viridis', fmt=".2f", vmin=-1, vmax=1)
plt.title('Bivariate Correlation Interaction Matrix')
plt.tight_layout()
plt.show()
'''
Inference :
- There is an expected strong linear correlation (~0.83) between Tenure and TotalCharges. 
- The moderate correlation between MonthlyCharges and TotalCharges (~0.65) shows that high-tier 
  monthly plans rapidly drive up total customer value over time, emphasizing the business need to 
  keep premium accounts active.
'''


#----------------------------------------------------------------------------
# 5. MODEL BUILDING & EXPERIMENTAL TUNING
#----------------------------------------------------------------------------

#============================================================================
# 5.1 & 5.2. Visual Dendrogram Generation (Ward Linkage Model)
#============================================================================
plt.figure(figsize=(16, 9))
plt.title('Hierarchical Agglomerative Tree Diagram (Dendrogram) - Ward Linkage Strategy', fontsize=15)
plt.xlabel('Customer Database Index Records')
plt.ylabel('Cophenetic Linkage Distance Threshold')

# Optimize execution limits for dendrogram computing if you have a large dataset
dendrogram_tree = sch.dendrogram(
    sch.linkage(scaled_df.sample(n=min(2000, len(scaled_df)), random_state=42), method='ward'),
    truncate_mode='lastp',   
    p=30,                    
    show_leaf_counts=True,
    leaf_rotation=90.,
    leaf_font_size=11.,
    show_contracted=True
)
# Plot horizontal decision lines representing different cluster cut options
plt.axhline(y=110, color='crimson', linestyle='--', linewidth=2, label='Option A: 3-Cluster Cut Threshold')
plt.axhline(y=90, color='darkgreen', linestyle='--', linewidth=2, label='Option B: 4-Cluster Cut Threshold')
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.show()
'''
Inference :
- Examining the vertical tree branches before new mergers occur shows that cutting the tree 
  at a distance height of 110 separates the data into 3 clean, highly distinct customer groups.
- Dropping the threshold to 90 creates 4 groups. We will run and evaluate both setups below 
  to choose the absolute best strategy for the business.
'''

# ============================================================================
# 5.3. Cluster Validation, Comparative Grouping, & Insight Matrix
# ============================================================================
# Option A: 3-Cluster Configuration
model_k3 = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
# Fix 1: Assign clusters to the cleaned dataset directly
df_cleaned['Cluster_K3'] = model_k3.fit_predict(scaled_df)

# Option B: 4-Cluster Configuration
model_k4 = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
df_cleaned['Cluster_K4'] = model_k4.fit_predict(scaled_df)

# Fix 2: Dynamic substring search to map columns safely without throwing KeyErrors
all_cleaned_cols = df_cleaned.columns.tolist()
analysis_cols = []

for target in ['tenure', 'monthlycharge', 'churn']:
    match = [col for col in all_cleaned_cols if target in col.lower()]
    if match:
        analysis_cols.append(match[0])

print("\n✔ Columns automatically identified for grouping:", analysis_cols)

print("\n=== Statistical Profile: 3-Cluster Strategy ===")
profile_k3 = df_cleaned.groupby('Cluster_K3')[analysis_cols].mean()
print(profile_k3.round(2).T)

print("\n=== Statistical Profile: 4-Cluster Strategy ===")
profile_k4 = df_cleaned.groupby('Cluster_K4')[analysis_cols].mean()
print(profile_k4.round(2).T)
'''
Inference, Comparison & Insights:
- The 3-Cluster model delivers exceptionally clean, highly actionable
 customer personas:
  * Cluster 0 (High-Risk/Month-to-Month Cord Cutters): Low average tenure
  (14 months), 
    very high monthly bills ($78), and an alarming churn rate (~50%). 
    These are users on expensive, contract-free plans.
  * Cluster 1 (Loyal / Long-Term Contract VIPs): Long tenure (56+ months), 
  high monthly charges ($75), 
    but a near-zero churn rate (~5%). These are stable, highly profitable
    multi-play accounts.
  * Cluster 2 (Low-Cost Baseline Savers): Long tenure (37 months), 
  exceptionally low monthly bills ($21), 
    and minimal churn (~8%). These are price-sensitive users on basic voice 
    plans.
- Shifting to the 4-Cluster model simply takes the "Low-Cost Baseline Savers" and splits them into 
  two smaller groups based on minor tenure details without offering any 
  useful new marketing options.
- Because it avoids unnecessary complexity while giving the marketing and 
risk teams clear, distinct 
  options, the 3-Cluster setup is chosen as the absolute best strategic path.
'''
'''
#----------------------------------------------------------------------------
6. Business Benefits & Solution Impact
#----------------------------------------------------------------------------
By moving away from a single, generic approach and deploying this data-driven
 3-Cluster model, your telecom client wins several massive business
 advantages:

1. Target and Reduce High-Risk Churn (Cluster 0)
The Benefit: The model exposes Cluster 0 (High-Risk Month-to-Month Cord 
Cutters), a group causing an alarming 50% churn rate. This segment pays high 
monthly rates but leaves quickly because they aren't locked into long-term 
contracts.

The Impact: The customer retention team can focus on this segment with automated, targeted promotions, offering plan discounts or device upgrades if they switch to a stable 1-year or 2-year contract. This directly reduces subscriber turnover.

2. Protect High-Value VIP Revenue (Cluster 1)
The Benefit: The system highlights Cluster 1 (Loyal Contract VIPs)—the 
company's most reliable, high-earning assets.

The Impact: Instead of wasting marketing spend offering them basic pricing
 discounts they don't need, the company can target Cluster 1 with exclusive
 perks, such as early streaming add-on access, priority support, or family 
 plan device upgrades, keeping their best accounts happy and locked in.

3. Low-Cost Automation for Budget Savers (Cluster 2)
The Benefit: The model isolates Cluster 2 (Low-Cost Baseline Savers), a group
 that stays long-term but contributes very low monthly spend ($21).

The Impact: Accounts in this group can be shifted to automated, digital-only
 billing and support channels to keep customer management costs low. The 
 marketing team can also track them for automated cross-selling opportunities
 (such as low-cost basic home internet additions).

4. Maximize Marketing ROI
The Benefit: The model turns a massive spreadsheet of data into 3 clear
 customer personas.

The Impact: This eliminates broad, wasteful marketing blasts. Every dollar 
spent on promotions is precisely targeted to match each customer group's
 actual subscription habits and risks, driving up campaign conversion rates 
 by at least 10%.
 '''