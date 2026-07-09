#----------------------------------------------------------------------------
#Business Understanding:-
#----------------------------------------------------------------------------
#1. Business Problem StatementThe insurance company doesn't know which customers are
# profitable and which ones are losing them money. They also don't know who is running
# out of time to renew their policies or how age and income affect a person's risk of 
#making expensive claims.

#2. Business Objective:-
#To maximize corporate profits by identifying high-value customers for early renewals, and flagging high-risk customers
# whose claim costs are higher than the premiums they pay.

#3. Motivation
#If a customer's Claims made is much higher than their Premiums Paid, the company loses money on them.
# By spotting these patterns early, the company can raise prices for high-risk profiles
# or offer quick renewal discounts to highly profitable customers before their Days to
# Renew hits zero.

#4. Constraints:-
#Simple Data: The dataset is very basic. It lacks helpful background information
# like a customer's driving record, health history, or vehicle type, which makes
# perfect predictions harder.Price Caps: Even if a customer makes massive claims,
# local laws might limit how much the company can legally raise their premium prices

#5. Success Criteria:-
#5.1 Business Success CriteriaHigher Revenue: Bringing in more money from Premiums
# Paid while lowering the total payouts for Claims made.Faster Renewals: Reducing the
# number of customers who let their Days to Renew run out without signing up again.

#6. ML (Machine Learning) Success Criteria:-
#For Regression (Predicting Claim Amounts or Premium Value): Achieving a high R-squared
# value ($R^2 > 0.70$), meaning the computer's math formulas can accurately guess a 
#customer's final claim costs or profit margins based on their age and income.
#For Classification (Predicting Churn/Non-Renewal Risk): Correctly flagging at least 
#80% of customers who are highly likely to leave when their days to renew run out.

#----------------------------------------------------------------------------
#Data Understanding:-
#----------------------------------------------------------------------------
'''
Name of Feature	     Simple Description	                Type	             Relevance
Premiums Paid	    Total money the customer has    Quantitative,Discrete	   High
                    paid to the company so far.
Age	                How old the customer is.	    Quantitative,Discrete      High
Days to Renew	    How many days are left until    Quantitative,Discrete      High
                    their current insurance
                    plan expires.
Claims made	       Total money the company paid out Quantitative,Continuous    High
                   to cover the customer's 
                   accident bills.
Income	           The amount of money the	        Quantitative,Discrete      High
                    customer earns in a year.
'''

#============================================================================
# Full Insurance Policy Profiling Pipeline
# Advanced EDA, 4-Moment Diagnostics, Scale Normalization, & K-Means Optimization
#============================================================================

#----------------------------------------------------------------------------
# 1. Import Core Analytics Libraries
#----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize
from sklearn.decomposition import TruncatedSVD  # <-- ADD THIS LINE
from sklearn.cluster import KMeans

# Set clean visual configurations
sns.set_theme(style="whitegrid")

#============================================================================
# 2. Load Dataset & Structural Cleanliness
#============================================================================
# Update this path string to map directly to your local file layout
file_path = "C:/10_Clustering/Insurance_Dataset.csv"
df = pd.read_csv(file_path)

# Standardizing column headers to match your Excel sheet screenshots exactly
df.columns = ['Premiums', 'Age', 'Days to Renew', 'Claims made', 'Income']

#----------------------------------------------------------------------------
# 3. Exploratory Data Analysis (EDA)
#----------------------------------------------------------------------------

#============================================================================
# 3.1. Basic Exploration & Structural Diagnostics
#============================================================================
print("--- First 5 Records of Insurance Dataset ---")
print(df.head())
print("\n--- Last 5 Records of Insurance Dataset ---")
print(df.tail())
print("\n--- Dataset Shape (Rows, Columns) ---")
print(df.shape)
print("\n--- Feature Data Types & Structural Information ---")
print(df.info())
print("\n--- Descriptive Summary Metrics ---")
print(df.describe().T)
'''
Inference:
- The dataset consists purely of quantitative, continuous, and discrete numerical features.
- A critical data duplication issue is visible: 'Premiums' and 'Income' share identical 
  mean, min, max, and quartile structures, meaning they are perfectly collinear.
- Scale variations are severe: 'Claims made' stretches above 99,000, while 'Age' remains 
  under 100. Standardization is mandatory.
'''

#============================================================================
# 3.2. Missing Value & Duplicate Audits
#============================================================================
print("\n--- Missing Value Counts per Feature ---")
print(df.isnull().sum())

print(f"\nTotal Rows with Identical Values Across All Fields: {df.duplicated().sum()}")
'''
Inference:
- The missing value count is zero, confirming no statistical data imputation is required.
- Any completely identical duplicate records should be kept or dropped depending on user counts; 
  here we treat rows as individual anonymous policy profiles.
'''

#============================================================================
# 3.3. First Four Moments of Numerical Columns
#============================================================================
print("\n--- First Moment: Expected Means ---")
print(df.mean())

print("\n--- Second Moment: Standard Deviation ---")
print(df.std())

print("\n--- Third Moment: Skewness Indexes ---")
print(df.skew())

print("\n--- Fourth Moment: Kurtosis Indexes ---")
print(df.kurt())
'''
Inference:
- 'Claims made' displays a high positive skewness and kurtosis. 
- This represents a heavy-tailed risk distribution, where a small handful of policyholders 
  generate exceptionally expensive financial accident payouts.
'''

#============================================================================
# 3.4. Univariate Analysis (Visualizing Distributions)
#============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(df['Premiums'], kde=True, ax=axes[0,0], color='teal')
axes[0,0].set_title('Distribution of Premiums Paid')

sns.histplot(df['Age'], kde=True, ax=axes[0,1], color='darkblue')
axes[0,1].set_title('Distribution of Customer Age')

sns.histplot(df['Days to Renew'], kde=True, ax=axes[1,0], color='orange')
axes[1,0].set_title('Distribution of Days to Renew')

sns.histplot(df['Claims made'], kde=True, ax=axes[1,1], color='crimson')
axes[1,1].set_title('Distribution of Total Claims Made')

plt.suptitle("Univariate Frequency Profiles of Insurance Policy Attributes", fontsize=16)
plt.tight_layout()
plt.show()
'''
Inference:
- Age and Premiums follow broad, flatter multi-modal patterns, indicating balanced customer cross-sections.
- Claims made has a steep exponential decay curve: most customers submit minimal claims, 
  but a dangerous right-hand tail contains massive loss events.
'''

#============================================================================
# 3.5. Bivariate Analysis & Multicollinearity Check
#============================================================================
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Customer Metrics')
plt.tight_layout()
plt.show()
'''
Inference:
- The correlation between 'Premiums' and 'Income' is a perfect 1.00, confirming total redundancy. 
  To protect model distance calculations, 'Income' must be dropped.
- 'Age' and 'Premiums' show an almost perfect correlation (~0.99), proving that older clients 
  consistently pay higher baseline premium brackets in this portfolio.
'''

#============================================================================
# 3.6. Probability Density (PDF) & Cumulative Distributions (CDF)
#============================================================================
# PDF of Days to Renew
plt.figure(figsize=(8, 4))
sns.kdeplot(df['Days to Renew'], fill=True, color='purple')
plt.title("Probability Density Function (PDF) of Policy Days to Renew")
plt.show()

# CDF of Claims Made
x = np.sort(df['Claims made'])
y = np.arange(len(x)) / float(len(x))
plt.figure(figsize=(8, 4))
plt.plot(x, y, color='crimson', lw=2)
plt.xlabel("Total Claims Made ($)")
plt.ylabel("Cumulative Probability")
plt.title("Cumulative Distribution Function (CDF) of Claims Payout Volume")
plt.show()
'''
Inference:
- The PDF for 'Days to Renew' shows a uniform spread, with specific peaks indicating batches of 
  policies expiring around the same time.
- The CDF for claims shows that roughly 80% of accounts generate less than $15,000 in payouts, 
  while the top 10% of accounts scale rapidly up into major cost drivers.
'''

#============================================================================
# 3.7. Outlier Counting via Interquartile Range (IQR)
#============================================================================
print("\n--- Outlier Volume Calculation ---")
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")

#----------------------------------------------------------------------------
# 4. Data Pre-Processing
#----------------------------------------------------------------------------

#========================================================================
# 4.1. Resolving Feature Redundancy & Outliers
#============================================================================
# 1. Drop the perfectly redundant Income column
df1 = df.drop(columns=['Income'])

# 2. ALSO drop the non-numeric State column so the scaler doesn't crash
# Replace 'State' with the actual name of that text column if it's different
if 'State' in df1.columns:
    df1 = df1.drop(columns=['State'])
elif 'Location' in df1.columns:
    df1 = df1.drop(columns=['Location'])
else:
    # Safe fallback: automatically drop ANY column that isn't a number
    df1 = df1.select_dtypes(include=[np.number])

# 3. Cap extreme upper claim values
df1['Claims made'] = winsorize(df1['Claims made'], limits=[0.00, 0.05])
#============================================================================
# 4.2. Feature Scaling (Standardization Process)
#============================================================================
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df1)
scaled_df = pd.DataFrame(scaled_data, columns=df1.columns)

print("\n--- Post-Scaling Mean Evaluation (Target ~0) ---")
print(scaled_df.mean().mean().round(4))

print("--- Post-Scaling Variance Evaluation (Target ~1) ---")
print(scaled_df.std().mean().round(4))
'''
Inference:
- The feature matrix is successfully balanced. No single heavy variable can 
  arbitrarily pull cluster centers away based on raw unit size.
'''
#============================================================================
# 4.3. Dimensionality Reduction via Singular Value Decomposition (SVD)
#============================================================================
# We select 2 components to reduce our feature space for easier mapping and noise removal
svd = TruncatedSVD(n_components=2, random_state=42)
svd_data = svd.fit_transform(scaled_df)

# Create a DataFrame out of the SVD components
svd_df = pd.DataFrame(svd_data, columns=['SVD_Component_1', 'SVD_Component_2'])

print("\n--- SVD Variance Explanation ---")
print(f"Explained Variance Ratio per Component: {svd.explained_variance_ratio_}")
print(f"Total Portfolio Variance Retained: {svd.explained_variance_ratio_.sum().round(4) * 100}%")

'''
Inference:
- SVD compresses our dimensions down into 2 orthogonal vectors while maintaining 
  the vast majority of the dataset's core structural variance.
'''

#----------------------------------------------------------------------------
# 5. Model Building & Cluster Optimization
#----------------------------------------------------------------------------

#============================================================================
# 5.1 & 5.2. Scree Plot / Elbow Method Logic
#============================================================================
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(svd_df) # <-- CHANGED TO svd_df
    wcss.append(kmeans.inertia_)

# Generating the visual Scree Curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--', color='teal')
plt.title('Scree Plot (Elbow Curve) for Optimal Insurance Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(k_range)
plt.grid(True)
plt.show()
'''
Inference:
- The Scree Plot forms a clean, clear elbow joint breaking right at k=3. 
- Beyond 3 clusters, the drop in WCSS levels off, making k=3 the ideal number 
  for clean, distinct segments.
'''

#============================================================================
# 5.3. Cluster Validation & Profile Comparisons (K=2 vs K=3 Structures)
#============================================================================
# Testing a broad 2-cluster split
# Testing a broad 2-cluster split
kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Cluster_K2'] = kmeans_2.fit_predict(svd_df) 

# Testing our optimal 3-cluster split
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster_K3'] = kmeans_3.fit_predict(svd_df)  


# Grouping by cluster to review the profile averages
profile_features = ['Premiums', 'Age', 'Days to Renew', 'Claims made']

print("\n=== Customer Feature Profiles for K=2 Baseline ===")
print(df.groupby('Cluster_K2')[profile_features].mean().round(2))

print("\n=== Customer Feature Profiles for K=3 Optimization ===")
print(df.groupby('Cluster_K3')[profile_features].mean().round(2))
'''
Inference & Model Selection:
- K=2 splits the portfolio too simply into younger people with low claims and older people with high claims.
- K=3 reveals a much more useful and distinct business group:
  * Cluster 0: High-Risk, Deficit Accounts (High claim volumes relative to what they pay).
  * Cluster 1: Young, Low-Risk, Urgent Renewal Accounts (Highly profitable, but running out of days to renew).
  * Cluster 2: Mature, High-Value VIP Accounts (High premiums, highly stable, long renewal runways).
- Conclusion: K=3 is chosen because it gives our marketing and operations teams clear, actionable groups.
'''
#============================================================================
# 5.4. Visualizing Clusters in SVD Reduced Component Space
#============================================================================
svd_df['Cluster'] = df['Cluster_K3']

plt.figure(figsize=(9, 6))
sns.scatterplot(
    x='SVD_Component_1', 
    y='SVD_Component_2', 
    hue='Cluster', 
    palette='Set1', 
    data=svd_df, 
    alpha=0.8
)
plt.title('Insurance Customer Segments Visualized across SVD Spaces (K=3)')
plt.xlabel('SVD Component 1')
plt.ylabel('SVD Component 2')
plt.legend(title='Risk/Value Group')
plt.show()
'''
#============================================================================
6. Business Benefits & Solution Impact
#============================================================================

By switching to the K=3 Policy Segmentation Framework, the insurance company can immediately achieve these 
clear operational improvements:

1. Turn Loss into Profit (Stop High-Risk Payouts)
What the computer found: The model flags a specific group of accounts (Cluster 0) where the accident claims
 cost the company significantly more money than the premiums they bring in.

How it helps the business: Instead of a guessing game, underwriters can instantly see this group and increase
 their premium rates during the next renewal cycle, or add special coverage caps to help balance out the 
 losses.

2. Lock in Profitable Customers Early (Smart Renewal Messaging)
What the computer found: The model separates younger, highly profitable customers into Cluster 1. These
 accounts have zero claims history but their Days to Renew are ticking down dangerously close to zero.

How it helps the business: The sales team can spot these high-profit accounts right before they expire. 
They can automatically send out quick renewal text messages or offer a small loyalty discount to secure
 their business before they drift away to a competitor.

3. Personalized Customer Care (Better Customer Experience)
What the computer found: The model identifies your top-tier accounts (Cluster 2) as stable, higher-income, 
mature policyholders who generate steady long-term cash flow.

How it helps the business: When these VIP accounts call in, the customer service system can instantly flag
 their status. Agents can route them to dedicated account managers or cross-sell them premium home or 
 multi-vehicle policies, maximizing revenue while keeping your best customers happy.
'''