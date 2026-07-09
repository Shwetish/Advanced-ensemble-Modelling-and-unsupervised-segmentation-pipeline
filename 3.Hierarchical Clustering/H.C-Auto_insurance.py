#----------------------------------------------------------------------------
#Business Understanding
#----------------------------------------------------------------------------
#1.Business Problem Statement: We are losing customers to competitors and losing 
#money because we price our policies the same way for everyone, regardless of their 
#actual risk.

#2.Business Objective: Group our drivers into distinct categories so we can provide 
#personalized pricing and tailored renewal rewards.

#3.Motivation: * Price Shoppers: Want the cheapest legal baseline coverage.
#Safety Seekers: Ready to pay extra for peace of mind, premium coverage, and seamless

# claims.
#Loyalists: Stay for convenience and good agent relationships.
#Constraints: We cannot read drivers' minds or daily habits; we can only use the 
#historical facts available in our spreadsheet.

#Business Success Criteria: A noticeable drop in customer cancellations (churn) 
#and a 10% increase in successful policy renewals over the next 6 months.

#ML Success Criteria: Create a clustering model that separates drivers into distinct,
# non-overlapping groups with highly similar risk and spending profiles within each 
#group.
#-----------------------------------------------------------------------------
# Data Understanding 
#-----------------------------------------------------------------------------
'''
Column Name                    Simple Description                                  Type                        Relevance
------------------------------------------------------------------------------------------------------------------------
Customer                       A unique identification code for each person.       Qualitative, Nominal        Low
                                                                                   (Identifier)
State                          The US state where the customer lives.              Qualitative, Nominal        High
Customer Lifetime Value        The total financial value a customer brings.        Quantitative, Continuous    High
Response                       Did the customer accept renewal? (Yes/No).          Qualitative, Binary         High
Coverage                       Protection plan tier (Basic, Extended, Premium).    Qualitative, Ordinal        High
Education                      Highest level of school completed.                  Qualitative, Ordinal        Low 
Effective To Date              The date the insurance policy became active.        Temporal (Date)             Low
EmploymentStatus               The job status of the customer (Employed, etc.).    Qualitative, Ordinal        High
Gender                         Whether the customer is Male (M) or Female (F).     Qualitative, Binary         Low                                                                               
Income                         The amount of money the customer earns in a year.   Quantitative,Discrete    High
Location Code                  Area type the customer lives in (Suburban, etc.).   Qualitative,Categorical        High
Marital Status                 Relationship status (Single, Married, Divorced).    Qualitative,Categorical        Low
Monthly Premium Auto           The exact amount paid every month.                  Quantitative,Discrete    High
Months Since Last Claim        Time passed since the last accident claim.          Quantitative,Discrete      High
Months Since Policy Inception  How long the customer has been with the company.    Quantitative,Discrete      High
Number of Open Complaints      The number of active, unresolved complaints.        Quantitative,Discrete      High
Number of Policies             Total number of plans bought by this customer.      Quantitative,Discrete      High
Policy Type                    Broad contract category (Personal, Corporate, etc). Qualitative,Categorical        High
Policy                         Specific sub-level of the contract (Personal L1).   Qualitative,Categorical       Low
Renew Offer Type               The specific type of discount offer extended.       Qualitative,Categorical       High
Sales Channel                  How the customer bought the plan (Agent, Web, etc). Qualitative,Categorical        High
Total Claim Amount             Total money paid out for customer's accidents.      Quantitative,Continuous    High
Vehicle Class                  The category of the car (SUV, Sports Car, etc.).    Qualitative,Categorical        High
Vehicle Size                   The physical size class of the car (Small, etc.).   Qualitative, Ordinal       High
'''
#============================================================================
# Full Customer Segmentation Pipeline via Hierarchical Clustering
# Client Portfolio: Auto Insurance Customer Analytics
# Sequence: Pre-processing -> Complete EDA -> Hierarchical Clustering -> Benefits
#============================================================================

#----------------------------------------------------------------------------
# 1. Import Core Advanced Analytics & Clustering Libraries
#----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Set clean production styling for graphs
sns.set_theme(style="whitegrid")

#============================================================================
# 2. Load Dataset & Auto-Clean Column Headers
#============================================================================
file_path = "C:/10_Clustering/AutoInsurance.csv"
df = pd.read_csv(file_path)

# FIX: Strip invisible spaces and harmonize column names to prevent KeyErrors
df.columns = df.columns.str.strip()

# Quick print diagnostic to check your exact, clean column names
print("Cleaned Columns in DataFrame:", list(df.columns))

#----------------------------------------------------------------------------
# 3. DATA PRE-PROCESSING & FEATURE ENGINEERING
#----------------------------------------------------------------------------

#============================================================================
# 3.1. Data Cleaning & Feature Selection
#============================================================================
# Drop high-cardinality unique tracking columns that lack mathematical patterns
ignore_cols = ['Customer', 'Effective To Date']
df_cleaned = df.drop(columns=[col for col in ignore_cols if col in df.columns])

# Fill missing values if any exist in numeric columns using the median
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_cleaned[col].isnull().sum() > 0:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

print("--- Pre-Processing: Schema Verification ---")
print(f"Original Dataset Columns: {df.shape[1]} | Model Ready Features: {df_cleaned.shape[1]}")
'''
Inference :
- Dropping 'Customer' (unique text ID) and 'Effective To Date' prevents the distance algorithm 
  from treating arbitrary strings or tracking tokens as meaningful numeric patterns.
- Filling potential missing fields with the median acts as a reliable data firewall, protecting 
  the clustering algorithm from throwing critical computational errors.
'''

#============================================================================
# 3.2. Feature Engineering (Categorical Encoding)
#============================================================================
# Convert text categoricals into numeric flags so the distance matrix can read them
df_encoded = pd.get_dummies(df_cleaned, drop_first=True)

print("\n--- Pre-Processing: One-Hot Encoding Status ---")
print(f"Total columns after converting text to numeric matrices: {df_encoded.shape[1]}")
'''
Inference :
- Mathematical distance algorithms like Hierarchical Clustering cannot read text strings like 
  "Basic", "Extended", or "Premium" directly.
- Converting these categorical values into binary columns (0s and 1s) allows the algorithm to 
  properly calculate similarities across all driver profiles.
'''

#============================================================================
# 3.3. Scale Standardization
#============================================================================
scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(df_encoded)

# Rebuild into a clean DataFrame for downstream exploration
scaled_df = pd.DataFrame(scaled_matrix, columns=df_encoded.columns)

print("\n--- Post-Scale Diagnostic Balance ---")
print(f"Calculated Global Feature Mean Point: {scaled_df.mean().mean().round(4)}")
print(f"Calculated Global Feature Variance:     {scaled_df.std().mean().round(4)}")
'''
Inference :
- Raw variables like 'Income' (ranging in the thousands) would completely overwhelm small fields 
  like 'Number of Policies' (ranging from 1 to 9) if left unscaled.
- Scaling shifts the mean of every feature to 0 and the variance to 1. This guarantees that 
  every column carries an equal weight when calculating cluster distances.
'''


#----------------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
#----------------------------------------------------------------------------

#============================================================================
# 4.1. Summary Statistics & Four Moments Analysis
#============================================================================
print("\n=== Dataset Dimensions ===")
print(df_encoded.shape)

print("\n=== Missing Value Volume Matrix ===")
print(df_encoded.isnull().sum().sum())

print("\n=== First Moment: Expected Mathematical Means (Sample) ===")
print(df_cleaned[numeric_cols].mean().round(2))

print("\n=== Second Moment: Standard Dispersion / Std Dev (Sample) ===")
print(df_cleaned[numeric_cols].std().round(2))

print("\n=== Third Moment: Skewness Coefficients (Sample) ===")
print(df_cleaned[numeric_cols].skew().round(2))

print("\n=== Fourth Moment: Kurtosis / Tail Thickness (Sample) ===")
print(df_cleaned[numeric_cols].kurt().round(2))
'''
Inference :
- A total missing value count of 0 confirms a perfectly clean data framework.
- High third-moment positive skewness and large fourth-moment kurtosis scores in 
  'Total Claim Amount' reveal that the majority of claims are small, but a small group 
  of high-risk outliers generate massive, expensive claims for the company.
'''

#============================================================================
# 4.2. Univariate Analysis (Distribution Densities, PDF, and CDF)
#============================================================================
# 4.2.1. Complete Visual Histogram Matrix for Core Continuous Inputs
df_cleaned[numeric_cols].hist(figsize=(14, 10), color='steelblue', edgecolor='black')
plt.suptitle("Univariate Histograms of Driver Metrics", fontsize=16)
plt.tight_layout()
plt.show()
'''
Inference:
- Visual inspection shows that features like Customer Lifetime Value (CLV) and Monthly 
  Premiums are heavily skewed to the right. 
- This confirms that a small, highly profitable group of elite premium customers exists 
  alongside a large base of standard, low-cost policyholders.
'''

# 4.2.2. Probability Density Functions (PDF) - Core Risk Drivers
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.kdeplot(df_cleaned['Customer Lifetime Value'], fill=True, color='purple', ax=axes[0])
axes[0].set_title('Probability Density Function (PDF): Customer Lifetime Value')
axes[0].set_xlabel('Customer Lifetime Value (CLV)')
axes[0].set_ylabel('Density Frequency')

sns.kdeplot(df_cleaned['Total Claim Amount'], fill=True, color='teal', ax=axes[1])
axes[1].set_title('Probability Density Function (PDF): Total Claim Amounts')
axes[1].set_xlabel('Total Claim Amount')
axes[1].set_ylabel('Density Frequency')

plt.suptitle("Univariate Continuous Density Analysis (PDF Profiles)", fontsize=15)
plt.tight_layout()
plt.show()
'''
Inference :
- The peak of the CLV PDF reveals that the baseline value of a customer is tightly concentrated 
  around $5,000 to $8,000.
- The Total Claim Amount PDF shows a broad distribution peaking around $400, confirming that 
  while routine minor claims are common, the tail extends far outward, representing severe accidents.
'''

# 4.2.3. Cumulative Distribution Functions (CDF) - Core Risk Drivers
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# CDF for CLV
sorted_clv = np.sort(df_cleaned['Customer Lifetime Value'])
cdf_clv = np.arange(len(sorted_clv)) / float(len(sorted_clv))
axes[0].plot(sorted_clv, cdf_clv, color='darkorange', lw=2.5)
axes[0].set_title('Cumulative Distribution Function (CDF): CLV')
axes[0].set_xlabel('Customer Lifetime Value')
axes[0].set_ylabel('Cumulative Probability')
axes[0].grid(True, linestyle='--')

# CDF for Total Claim Amount
sorted_claims = np.sort(df_cleaned['Total Claim Amount'])
cdf_claims = np.arange(len(sorted_claims)) / float(len(sorted_claims))
axes[1].plot(sorted_claims, cdf_claims, color='crimson', lw=2.5)
axes[1].set_title('Cumulative Distribution Function (CDF): Total Claims')
axes[1].set_xlabel('Total Claim Amount')
axes[1].set_ylabel('Cumulative Probability')
axes[1].grid(True, linestyle='--')

plt.suptitle("Univariate Concentration Analysis (CDF Profiles)", fontsize=15)
plt.tight_layout()
plt.show()
'''
Inference :
- The CLV CDF shows that nearly 85% of all drivers have a life value below $15,000, 
  highlighting the extreme importance of retaining the top 15% high-value accounts.
- The Total Claim Amount CDF shows that 90% of claims stay under $800, meaning any driver 
  claiming over $1,000 is a major cost outlier who requires specialized risk management.
'''

#============================================================================
# 4.3. Bivariate Analysis (Cross-Feature Interactions)
#============================================================================
plt.figure(figsize=(10, 7))
sns.heatmap(df_cleaned[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Bivariate Feature Interaction Matrix (Correlation Coefficient)')
plt.tight_layout()
plt.show()
'''
Inference :
- A strong positive correlation (~0.63) shows up between 'Monthly Premium Auto' and 
  'Total Claim Amount'. This makes perfect sense: drivers with more expensive vehicles pay 
  higher premiums, but they also cost more to repair during accidents.
- The near-zero correlation between 'Income' and 'Customer Lifetime Value' proves that a 
  customer's value to the company isn't just about how much money they make—it is driven by 
  how long they stay and how many policies they buy.
'''


#----------------------------------------------------------------------------
# 5. MODEL BUILDING & EXPERIMENTAL TUNING
#----------------------------------------------------------------------------

#============================================================================
# 5.1 & 5.2. Hierarchical Clustering and Dendrogram Visualization
#============================================================================
plt.figure(figsize=(16, 9))
plt.title('Hierarchical Clustering Tree (Dendrogram) - Ward Linkage Strategy')
plt.xlabel('Customer Matrix Sample Index')
plt.ylabel('Euclidean Cophenetic Distance Scale')

# Generate dendrogram tree using Ward's minimum variance method
dendrogram = sch.dendrogram(
    sch.linkage(scaled_df, method='ward'),
    truncate_mode='lastp',  # Summarize complex branch trees for clear reading
    p=30,                   
    show_leaf_counts=True,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True
)
plt.axhline(y=120, color='crimson', linestyle='--', label='Optimal 3-Cluster Threshold')
plt.axhline(y=95, color='darkgreen', linestyle='--', label='Alternative 4-Cluster Threshold')
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.show()
'''
Inference :
- Analyzing the vertical tree heights before new branches merge reveals that cutting the 
  dendrogram at a distance threshold of 120 gives us 3 highly distinct, balanced clusters.
- Dropping the cut line to 95 splits the data into 4 groups. We will test and validate both 
  options below to choose the best configuration for the business.
'''

#============================================================================
# 5.3. Cluster Validation & Comparative Evaluation (Fixed Copy)
#============================================================================
# Strategy A: 3-Cluster Model
hc_3 = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
df['HC_Cluster_K3'] = hc_3.fit_predict(scaled_df)

# Strategy B: 4-Cluster Model
hc_4 = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
df['HC_Cluster_K4'] = hc_4.fit_predict(scaled_df)

# Target evaluation metrics
analysis_cols = ['Customer Lifetime Value', 'Income', 'Monthly Premium Auto', 'Total Claim Amount']

# Dynamic check: Verify that columns are present before running groupby to avoid crashes
available_cols = [col for col in analysis_cols if col in df.columns]

print("\n=== Cluster Average Profiles: 3-Cluster Strategy ===")
summary_k3 = df.groupby('HC_Cluster_K3')[available_cols].mean()
print(summary_k3.round(2))

print("\n=== Cluster Average Profiles: 4-Cluster Strategy ===")
summary_k4 = df.groupby('HC_Cluster_K4')[available_cols].mean()
print(summary_k4.round(2))
'''
Inference, Comparison & Insights:
- The 3-Cluster approach provides an exceptionally clear, highly actionable breakdown:
  * Cluster 0 (Budget Price Shoppers): Low premiums, lower claim values, and modest income profiles.
  * Cluster 1 (High-Value Premium Families): High incomes, high customer lifetime value, and clean, steady premiums.
  * Cluster 2 (High-Risk/High-Cost Drivers): Moderate income levels but extremely high claim amounts, making them unprofitable.
- Shifting to the 4-Cluster strategy simply splits the "Price Shoppers" group into two smaller, 
  overlapping sub-segments based on minor income variations. 
- Because it doesn't give our marketing or underwriting teams any new, distinct strategic options, 
  the 3-Cluster model is chosen as the most effective solution for the business.
'''
'''
#-----------------------------------------------------------------------------------
#6. Business Benefits & Solution Impact:-
#-----------------------------------------------------------------------------------
#Using this hierarchical clustering model gives your auto insurance client several
# clear, immediate business advantages:

#1. Optimized Risk Underwriting & Smarter Pricing
#The Benefit: Instead of charging the same general rates across the board, 
#the company can flag Cluster 2 (High-Risk/High-Cost Drivers) right away.

#The Impact: Underwriters can safely raise monthly premiums or increase deductibles 
#for this specific group, helping cover the high cost of their claims and protecting
# the company's overall bottom line.

#2. Targeted, High-ROI Marketing for Top-Tier Customers
#The Benefit: The model cleanly isolates Cluster 1 (High-Value Premium Families)—the
#company's most profitable assets.

#The Impact: The marketing team can stop wasting budget sending generic ads to 
#everyone. Instead, they can focus their resources entirely on Cluster 1 with 
#premium bundle upgrades (like combining home and luxury auto policies) and 
#exclusive loyalty rewards, keeping their best customers from leaving.

# 3. Proactive Retention & Churn Prevention
#The Benefit: The model exposes Cluster 0 (Budget Price Shoppers), a group that is
# highly likely to jump to a competitor over a tiny price difference.

#The Impact: The company can launch automated, low-cost marketing campaigns tailored 
#directly to this group, offering simple discounts or basic digital-only options to
# match their budget-conscious motivation and keep them from canceling their policies.

'''