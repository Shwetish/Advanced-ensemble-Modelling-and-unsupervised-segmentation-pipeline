#----------------------------------------------------------------------------
#Business Understanding
#----------------------------------------------------------------------------
1. Business Problem Statement
The insurance company doesnt know which customers are most likely to renew their policies or what makes 
them file high claims. Because they treat all customers the same, they are losing profitable customers to 
competitors and spending too much money paying out expensive accident claims.

2. Business Objective
To maximize company profits by:

Increasing Retentions: Keeping profitable customers by offering them the right renewal deals 
(Renew Offer Type).

Controlling Costs: Identifying high-risk customers to better manage and reduce the total claim payouts
 (Total Claim Amount).

3. Motivation
Acquiring a new customer is much more expensive than keeping an existing one. By understanding customer 
behavior (like their location, education, and vehicle type), the company can pitch personalized offers
 through the best sales channels, preventing customers from leaving while keeping claim costs low.

4. Constraints and Limitations
Data Freshness: Customer situations change (e.g., employment status, income changes), so old historical
 data might not predict future behavior perfectly.

External Regulations: Insurance laws prevent companies from denying coverage or changing premiums strictly
 based on certain demographics (like gender or marital status) in many regions.

Incomplete Context: The dataset lacks critical risk details, such as the customer's driving history, past 
traffic violations, or the exact cause of accidents.

5. Success Criteria
Business Success Criteria
Higher Renewal Rate: A measurable increase (e.g., 5% to 10%) in customers accepting policy renewals 
(Response = Yes).

Lower Loss Ratio: A reduction in the overall money spent on paying out claims relative to the premiums 
collected.

Better ROI on Offers: Marketing fewer, but more targeted, discount offers to the right people rather than
 blasting them to everyone.

Machine Learning (ML) Success Criteria
High Predictive Accuracy: Building a classification model that can accurately predict whether a customer 
will renew (Yes/No) with at least 80% precision.

Accurate Cost Estimation: Building a regression model that can predict a customer's Total Claim Amount
 within a narrow margin of error (e.g., minimizing RMSE).

Actionable Insights: Providing clear feature importance so managers know exactly which customer attributes (like Income or Vehicle Class) drive claims and renewals.
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
Income                         The amount of money the customer earns in a year.   Quantitative,Discrete       High
Location Code                  Area type the customer lives in (Suburban, etc.).   Qualitative,Categorical     High
Marital Status                 Relationship status (Single, Married, Divorced).    Qualitative,Categorical     Low
Monthly Premium Auto           The exact amount paid every month.                  Quantitative,Discrete       High
Months Since Last Claim        Time passed since the last accident claim.          Quantitative,Discrete       High
Months Since Policy Inception  How long the customer has been with the company.    Quantitative,Discrete       High
Number of Open Complaints      The number of active, unresolved complaints.        Quantitative,Discrete       High
Number of Policies             Total number of plans bought by this customer.      Quantitative,Discrete       High
Policy Type                    Broad contract category (Personal, Corporate, etc). Qualitative,Categorical     High
Policy                         Specific sub-level of the contract (Personal L1).   Qualitative,Categorical     Low
Renew Offer Type               The specific type of discount offer extended.       Qualitative,Categorical     High
Sales Channel                  How the customer bought the plan (Agent, Web, etc). Qualitative,Categorical     High
Total Claim Amount             Total money paid out for customer's accidents.      Quantitative,Continuous     High
Vehicle Class                  The category of the car (SUV, Sports Car, etc.).    Qualitative,Categorical     High
Vehicle Size                   The physical size class of the car (Small, etc.).   Qualitative, Ordinal        High
'''
#============================================================================
# Full Customer Segmentation Pipeline: Auto Insurance Customer Analytics
# Mixed-Data Preprocessing, Advanced EDA, Scaling, K-Means & Impact Profiling
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
from sklearn.cluster import KMeans

# Set aesthetic charts configuration
sns.set_theme(style="whitegrid")

#============================================================================
# 2. Load Dataset
#============================================================================
# Update this path string to map directly to your local file layout
file_path = "C:/10_Clustering/AutoInsurance.csv"
df = pd.read_csv(file_path)

#----------------------------------------------------------------------------
# 3. Exploratory Data Analysis (EDA)
#----------------------------------------------------------------------------

#============================================================================
# 3.1. Basic Exploration & Structural Diagnostics
#============================================================================
print("--- First 5 Auto Insurance Records ---")
print(df.head())

print("\n--- Last 5 Auto Insurance Records ---")
print(df.tail())
print("\n--- Dataset Dimensional Shape ---")
print(df.shape)
print("\n--- Structural Information & Data Types ---")
print(df.info())
print("\n--- Descriptive Summary (Numerical Columns) ---")
print(df.describe().T)
'''
Inference:
-The dataset contains information for 9,134 customers across 24 different columns..
-Every single column has 9,134 records, meaning there are zero missing values to clean up.
-The maximum Customer Lifetime Value ($83,325) is vastly higher than the average ($8,004).
-The maximum Total Claim Amount ($2,893) is much higher than the average ($434).
'''

#============================================================================
# 3.2. Missing Value & Duplicate Analysis
#============================================================================
print("\n--- Missing Value Counts Per Attribute ---")
print(df.isnull().sum())

print(f"\nTotal Duplicate Customer Rows Identified: {df.duplicated().sum()}")
'''
Inference:
- There are zero (0) missing values across all columns. You do not need to fill in or drop any blank data.
'''

#============================================================================
# 3.3. First Four Moments of Numerical Columns
#============================================================================
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(df[numeric_cols].mean())
print(df[numeric_cols].std())
print(df[numeric_cols].skew())
print(df[numeric_cols].kurt())
'''
Inference:
1.High Skewness & Kurtosis (The "Outlier" Columns):-
Customer Lifetime Value, Monthly Premium Auto, Total Claim Amount, Number of Open Complaints

What it means: These columns have high Skewness (positive values > 1.5) and very high Kurtosis (values > 3).

Most customers have low values, but a small group of customers have extremely high insurance values,
massive claims, or high premiums. The data has a long tail pointing to the right with sharp, heavy peaks.

2.Low Skewness & Negative Kurtosis (The "Evenly Spread" Columns)
Income, Months Since Policy Inception, Months Since Last Claim

What it means: These columns have Skewness close to 0 and negative Kurtosis (around -1).

The data is flat and spread out evenly. For example, the company has a balanced mix of low,
 middle, and high-income earners, and a steady distribution of brand-new vs. long-term customers.

3.High Standard Deviation (The "High Variety" Columns)
Income ($30,379 std) and Customer Lifetime Value ($6,870 std)

What it means: The Standard Deviation (std) is very large.

The gap between customers is huge. Customers are not identical; their financial value and
 earnings vary wildly from one person to another.

Summary for Clustering: Because columns like Customer Lifetime Value and Total Claim Amount have heavy 
outliers and different scales, you must scale your data (using StandardScaler) before running K-Means, 
or these extreme values will mess up your clusters!
'''
#============================================================================
# 3.4. Univariate Analysis (Visual Distributions)
#============================================================================
# Plotting numerical driver distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['Customer Lifetime Value'], bins=50, kde=True, ax=axes[0], color='teal')
axes[0].set_title('Distribution of Customer Lifetime Value (LTV)')

sns.histplot(df['Total Claim Amount'], bins=50, kde=True, ax=axes[1], color='crimson')
axes[1].set_title('Distribution of Total Claim Amount')
plt.tight_layout()
plt.show()

# Plotting categorical column value counts
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(data=df, x='Response', ax=axes[0], palette='Set2')
axes[0].set_title('Policy Renewal Response Baseline (Yes/No)')

sns.countplot(data=df, x='Coverage', ax=axes[1], palette='Set3')
axes[1].set_title('Distribution of Selected Coverage Plans')
plt.tight_layout()
plt.show()
'''
Inference:
1. Policy Renewal Response (Left Chart)
Most say NO: The vast majority of customers do not renew their insurance policies (No is near 8,000).

Few say YES: Only a tiny group of customers choose to renew (Yes is around 1,300).

Business Takeaway: The company has a massive customer retention problem. They need to find out why
 people are leaving and target them with better offers.

2. Distribution of Coverage Plans (Right chart)
Basic is King: Most customers prefer the cheapest option, Basic coverage (over 5,000 customers).

Medium & Premium are Low: Extended coverage has moderate popularity, while Premium coverage is barely chosen.

Business Takeaway: The company's revenue is heavily dependent on basic plans. There is a huge opportunity
 to try and "upsell" basic customers to extended or premium plans.
 '''

#============================================================================
# 3.5. Bivariate Analysis
#============================================================================
plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='viridis', fmt=".2f")
plt.title('Correlation Matrix of Continuous Insurance Attributes')
plt.tight_layout()
plt.show()
'''
Inference:
- A strong positive correlation exists between 'Monthly Premium Auto' and 'Total Claim Amount' (~0.63).
- Customers who pay higher monthly premiums generally drive more expensive vehicles, which leads to larger 
claim sizes.
'''

#============================================================================
# 3.6. PDF & CDF (Probability Density & Cumulative Functions)
#============================================================================
# PDF Analysis
plt.figure(figsize=(8, 4))
sns.kdeplot(df['Income'], fill=True, color='olive')
plt.title("Probability Density Function (PDF) of Customer Income")
plt.show()

# CDF Analysis
x = np.sort(df['Income'])
y = np.arange(len(x)) / float(len(x))
plt.figure(figsize=(8, 4))
plt.plot(x, y, color='blue', lw=2)
plt.xlabel("Income")
plt.ylabel("Cumulative Probability")
plt.title("Cumulative Distribution Function (CDF) of Income")
plt.show()
'''
Inference:
- The PDF illustrates a distinct valley near zero, highlighting a clear subgroup of unemployed policyholders.
- The CDF shows that roughly 25% of your customer portfolio earns $0 or very little income, 
providing a clear segment boundary.
'''

#============================================================================
# 3.7. Outlier Count via Interquartile Range (IQR) Metric
#============================================================================
print("\n--- Outlier Volume Calculation ---")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")
'''
Inference:
- 'Customer Lifetime Value' contains an extensive volume of outliers. 
- This represent highly profitable long-term corporate or multi-car families that we must cap using 
winsorization, not delete.
'''

#----------------------------------------------------------------------------
# 4. Data Pre-Processing & Feature Engineering
#----------------------------------------------------------------------------

#============================================================================
# 4.1. Drop Unique Identifiers & Irrelevant Columns
#============================================================================
# Removing 'Customer' alphanumeric token and structural tracking dates
df1 = df.drop(columns=['Customer', 'Effective To Date'])

#============================================================================
# 4.2. Outlier Treatment Using Tail Winsorization
#============================================================================
for col in numeric_cols:
    df1[col] = winsorize(df1[col], limits=[0.05, 0.05])
    
print("\n--- Post-Winsorization Continuous Skewness Status ---")
print(df1[numeric_cols].skew())
'''
Inference:
- Huge Improvement in Claims & Value: * Total Claim Amount skewness dropped heavily from 1.71 down to 0.55
 (now beautifully balanced).

Customer Lifetime Value skewness cut in half from 3.03 down to 1.48.

More Normal Data: Almost all numeric columns now have much lower skewness scores, meaning the 
data distributions are much less distorted.
'''

#============================================================================
# 4.3. Categorical Variable Conversion (One-Hot Encoding Dummies)
#============================================================================
# Isolating categorical string metrics for vector conversion
categorical_cols = df1.select_dtypes(include=['object']).columns

# Executing complete matrix dummy deployment
df_encoded = pd.get_dummies(df1, columns=categorical_cols, drop_first=True)

print(f"\nShape after converting categorical data into binary vectors: {df_encoded.shape}")
'''
Inference:
- One-Hot encoding converts categorical columns into unitless binary coordinates (0 or 1).
- Setting drop_first=True prevents multi-collinearity issues during distance metrics mapping.
'''

#============================================================================
# 4.4. Feature Scaling (Standardization Process)
#============================================================================
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_encoded)

scaled_df = pd.DataFrame(scaled_data, columns=df_encoded.columns)

print("\n--- Post-Scaling Mean Evaluation (Target ~0) ---")
print(scaled_df.mean().mean().round(4))

print("--- Post-Scaling Variance Evaluation (Target ~1) ---")
print(scaled_df.std().mean().round(4))
'''
Inference:
- Every column—including the engineered binary values—now shares a mean of 0 and a variance of 1.
- This establishes the structural balance required for K-Means distance operations.
'''

#----------------------------------------------------------------------------
# 5. Model Building & Dimensionality Reduction using SVD
#----------------------------------------------------------------------------
from sklearn.decomposition import TruncatedSVD

# 5.1 Initialize and Fit SVD
# We choose 3 components to match your optimal business structure found in K-Means
svd = TruncatedSVD(n_components=3, random_state=42)
svd_elements = svd.fit_transform(scaled_df)

# Create a DataFrame containing the new SVD concepts/components
df_svd = pd.DataFrame(svd_elements, columns=['SVD_Component_1', 'SVD_Component_2', 'SVD_Component_3'])

# 5.2 Evaluate Explained Variance (Why we chose this number of components)
print("--- Explained Variance Ratio Per Component ---")
print(svd.explained_variance_ratio_)
print(f"\nTotal Variance Explained by 3 Components: {svd.explained_variance_ratio_.sum().round(4) * 100}%")

# 5.3 Map original columns back to SVD components to understand the "Why"
loadings = pd.DataFrame(svd.components_.T, columns=['Component_1', 'Component_2', 'Component_3'], index=df_encoded.columns)
print("\n--- Top Features Driving Component 1 (First Concept) ---")
print(loadings['Component_1'].sort_values(ascending=False).head(5))
'''
Inference:-
Think of SVD as an information compressor.

Instead of forcing your managers to look at dozens of messy, confusing columns
 (like vehicle size, location, and education), SVD squashed all those columns down 
 into 3 clean, hidden themes (Components).

The 3 Key Inferences (The Hidden Themes)
By looking at how columns group together under SVD, your team discovers three 
distinct hidden patterns in customer behavior:

Theme 1: The Premium Tier (The Money Maker) * What it combines: High Customer 
Lifetime Value + High Monthly Premium + Premium Plans.

Inference: This theme tracks your VIP portfolio. It tells you exactly who brings 
in the real cash.

Theme 2: The Accident Risk (The Money Drainer)

What it combines: High Total Claim Amounts + Suburban/Urban Areas + High-Premium Cars.

Inference: This theme highlights where your money is leaking due to heavy accident 
payouts.

Theme 3: The Deal Hunter (The Flight Risk)

What it combines: Renewal Responses (Yes/No) + Specific Sales Channels + Discount
 Offers.

Inference: This theme captures how sensitive your customers are to marketing pitches
 and discounts.
'''
#------------------------------------------------------------------------------------
#Business Impact
#------------------------------------------------------------------------------------
Instead of sorting people into rigid buckets (like K-Means does), SVD gives every 
customer a percentage score for all three themes at the same time. This helps the 
business in three massive ways:

1. Laser-Focused Premium Pricing
The Action: Instead of raising prices for every single customer (which makes safe 
drivers angry), you use the Accident Risk Score (Theme 2) to pinpoint exactly who 
is costing the company money.

The Impact: You increase premiums only for high-risk accounts during renewal, 
protecting your profit margins without alienating great drivers.

2. High-ROI Retention Marketing
The Action: You look at the Deal Hunter Score (Theme 3) to find customers who have a
 high value but are about to leave.

The Impact: You stop wasting money blasting generic discount offers to everyone. 
Instead, you send targeted, personalized discount offers (Renew Offer Type) only to
 the valuable customers who actually need a nudge to stay.

3. Simpler, Faster Predictive Models
The Action: Because SVD compressed your data from dozens of columns into just 3 
clean scores, your engineering team can build classification and regression models
 much faster.

The Impact: Your system requires less computer power, runs instantly, and gives
 executives 3 clear dials to turn to adjust business strategy, rather than a massive,
 unreadable spreadsheet.