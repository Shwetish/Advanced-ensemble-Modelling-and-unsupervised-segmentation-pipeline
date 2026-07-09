#----------------------------------------------------------------------------
#Business Understanding
#----------------------------------------------------------------------------
#1. Business Problem Statement
#"Government funds, police officers, and safety resources are being divided equally 
#across all states, which wastes money in peaceful areas and leaves high-crime zones
# dangerously underfunded."

#2. Business Objective
#Group states into distinct risk clusters (like "Safe Tiers" or "High Violence Tiers")
# based on their actual crime statistics so policymakers can send resources exactly 
#where they are needed most.

#3. Motivation (The "Why")
#High-Risk Zones: Need urgent, heavy funding for violent crime tactical squads and
# crisis response.

#Moderate Property/Urban Crime Zones: Need more community policing, patrolling, and 
#social programs to stop crime from growing.

#Low-Crime/Safe Zones: Need minimal extra budget—just standard funds to maintain their
# current safety level.

#4. Constraints
#Tight Budgets: Government money is strictly limited; we cannot give maximum funding to
# every single state.

#Data Limits: The dataset only tracks three types of violent crimes and city population
# sizes. It lacks details on police presence, local laws, or economic status.

#5. Business Success Criteria
#Successfully group all 50 states into 3 to 4 easy-to-understand safety tiers that law 
#enforcement can actually use for budgeting.

#Help achieve a measurable drop in overall violent crimes (like assault and murder)
# over the next 2 years due to smarter resource placement.

#6. ML Success Criteria
#Create a clustering model (such as Hierarchical or K-Means) where states in the same 
#group have nearly identical crime rates, creating completely distinct, non-overlapping
# safety categories.
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
# Full Mixed-Data Customer Segmentation Pipeline via Hierarchical Clustering
# Client Portfolio: Auto Insurance Operations & Marketing Strategy
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
# Read the target insurance.csv file 
file_path = "C:/10_Clustering/insurance Dataset.csv"
df = pd.read_csv(file_path)


# Add this line to see the exact column names:
print("Actual Columns in CSV:", df.columns.tolist())

#----------------------------------------------------------------------------
# 3. DATA PRE-PROCESSING & FEATURE ENGINEERING
#----------------------------------------------------------------------------

#============================================================================
# 3.1. Data Cleaning & Drop Irrelevant String Trackers
#============================================================================
# Identify and remove high-cardinality nominal text identifiers (like tracking IDs) 
# that do not contain statistical patterns or distance values.
id_cols = ['Customer', 'Policy', 'Effective To Date']
df_cleaned = df.drop(columns=[col for col in id_cols if col in df.columns], errors='ignore')
# Automatic data firewall: Check for missing data and fill with the median (numeric) or mode (categorical)
for col in df_cleaned.columns:
    if df_cleaned[col].isnull().sum() > 0:
        if df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
        else:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

print("--- Pre-Processing: Cleaning & Structural Check ---")
print(f"Original Raw Schema Columns: {df.shape[1]} | Model-Ready Clean Columns: {df_cleaned.shape[1]}")
'''
Inference :
- Removing strict identifiers like 'Customer ID' prevents the distance matrix from treating 
  arbitrary text labels as real patterns.
- Confirming zero null fields ensures that the downstream mathematical distance equations 
  (Euclidean/Ward) can run without throwing fatal structural execution errors.
'''

#============================================================================
# 3.2. Feature Engineering (Converting Mixed Categorical to Numeric via Dummy Encoding)
#============================================================================
# Machine learning algorithms cannot calculate distance on text strings directly.
# We apply one-hot encoding to all categorical text columns to build a purely numeric matrix.
df_encoded = pd.get_dummies(df_cleaned, drop_first=True).astype(int)

print("\n--- Pre-Processing: One-Hot Dummy Encoding Matrix ---")
print(f"Total structured columns after expanding mixed text categories: {df_encoded.shape[1]}")
'''
Inference:
- Features like 'State', 'Coverage', and 'EmploymentStatus' are expanded into clean binary flags (0 or 1).
- Setting 'drop_first=True' eliminates multi-collinearity issues (the dummy variable trap), 
  ensuring that redundant duplicate features do not skew the distance weights.
'''

#============================================================================
# 3.3. Feature Normalization (MinMax Scaling Technique)
#============================================================================
# Because the data contains both binary flags (0 or 1) and massive scale numbers like Income,
# MinMax Normalization is selected to bound all metrics tightly between 0 and 1.
scaler = MinMaxScaler()
scaled_matrix = scaler.fit_transform(df_encoded)

# Re-build into a clean, structured DataFrame for EDA exploration
scaled_df = pd.DataFrame(scaled_matrix, columns=df_encoded.columns)

print("\n--- Pre-Processing: Normalization Scaling Diagnostic ---")
print(f"Global Scaled Minimum Bound: {scaled_df.min().min()} | Global Scaled Maximum Bound: {scaled_df.max().max()}")
'''
Inference :
- Before scaling, an 'Income' value of $80,000 would completely overwhelm a 'Number of Policies' value of 3.
- Normalizing compresses every single feature into an identical range of [0, 1]. This ensures 
  mixed attributes carry equal mathematical importance when determining customer similarity.
'''


#----------------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
#----------------------------------------------------------------------------

#============================================================================
# 4.1. Dataset Summary & Core Statistical Structure
#============================================================================
print("\n=== Data Dimensions Matrix ===")
print(df_encoded.shape)

# Select primary numerical categories for standard statistical observation
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
- The base average Customer Lifetime Value (CLV) is roughly $8,004.91, while the standard deviation is a large $6,901.
- High positive skewness (2.69) and kurtosis (7.34) values in CLV prove that the customer base 
  has a heavy right tail—meaning a small group of high-value patrons generate the vast majority of profit.
'''

#============================================================================
# 4.2. Univariate Analysis (Distribution Plots, PDF, and CDF Profiles)
#============================================================================
# 4.2.1. Complete Visual Histogram Array
df_cleaned[numeric_features].hist(figsize=(14, 10), color='cadetblue', edgecolor='black')
plt.suptitle("Univariate Frequency Histograms of Auto Insurance Metrics", fontsize=15)
plt.tight_layout()
plt.show()
'''
Inference :
- 'Income' displays a uniform, flat distribution across multiple wealth brackets, with a noticeable spike at 0 
  representing unemployed policyholders.
- 'Total Claim Amount' and 'Monthly Premium Auto' follow a severe right-hand skew, highlighting 
  common minor expenses alongside sparse, heavy claims.
'''

# 4.2.2. Continuous Probability Density Functions (PDF) - Core Segmentation Anchors
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.kdeplot(df_cleaned['Customer Lifetime Value'], fill=True, color='darkblue', ax=axes[0])
axes[0].set_title('Probability Density Function (PDF): Customer Lifetime Value')
axes[0].set_xlabel('CLV Range Scale')
axes[0].set_ylabel('Density Concentration')

sns.kdeplot(df_cleaned['Monthly Premium Auto'], fill=True, color='darkgreen', ax=axes[1])
axes[1].set_title('Probability Density Function (PDF): Monthly Premium Rates')
axes[1].set_xlabel('Monthly Premium Cost ($)')
axes[1].set_ylabel('Density Concentration')

plt.suptitle("Continuous Density Profiling (PDF)", fontsize=14)
plt.tight_layout()
plt.show()
'''
Inference:
- The CLV PDF peaks sharply around the $5,000 range, confirming where the basic risk-pool lives.
- The Monthly Premium PDF shows a massive peak near $70-$90, indicating that basic economy 
  auto insurance is the primary product seller for this business.
'''

# 4.2.3. Cumulative Distribution Functions (CDF) - Value Concentration Profiles
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# CDF calculation for CLV
sorted_clv = np.sort(df_cleaned['Customer Lifetime Value'])
cdf_clv = np.arange(len(sorted_clv)) / float(len(sorted_clv))
axes[0].plot(sorted_clv, cdf_clv, color='purple', lw=2.5)
axes[0].set_title('Cumulative Distribution Function (CDF): CLV')
axes[0].set_xlabel('Customer Lifetime Value ($)')
axes[0].set_ylabel('Cumulative Percentage Coverage')
axes[0].grid(True, linestyle=':')

# CDF calculation for Total Claims
sorted_claims = np.sort(df_cleaned['Total Claim Amount'])
cdf_claims = np.arange(len(sorted_claims)) / float(len(sorted_claims))
axes[1].plot(sorted_claims, cdf_claims, color='orangered', lw=2.5)
axes[1].set_title('Cumulative Distribution Function (CDF): Total Claims')
axes[1].set_xlabel('Total Claim Amount ($)')
axes[1].set_ylabel('Cumulative Percentage Coverage')
axes[1].grid(True, linestyle=':')

plt.suptitle("Cumulative Concentration Profiling (CDF)", fontsize=14)
plt.tight_layout()
plt.show()
'''
Inference :
- The CLV CDF shows that almost 80% of customers have an accumulated lifetime value 
below $10,000. 
- The Total Claims CDF demonstrates that 90% of all recorded insurance losses sit 
below $800, 
  which visually separates normal, safe drivers from high-loss outlier accounts.
'''

#============================================================================
# 4.3. Bivariate Analysis (Cross-Feature Dynamic Relationships)
#============================================================================
plt.figure(figsize=(9, 7))
sns.heatmap(df_cleaned[numeric_features].corr(), annot=True, cmap='viridis', fmt=".2f", vmin=-1, vmax=1)
plt.title('Bivariate Correlation Interaction Matrix')
plt.tight_layout()
plt.show()
'''
Inference :
- A strong positive linear correlation (~0.63) exists between 'Monthly Premium Auto'
 and 'Total Claim Amount'. 
  This indicates that high-premium vehicles naturally incur higher dollar-value 
  accident repairs.
- The weak correlation between 'Months Since Policy Inception' and CLV shows that a
 client's financial 
  value is driven more by active multi-car policies and cross-shopping than by sheer
  calendar tenure.
'''


#----------------------------------------------------------------------------
# 5. MODEL BUILDING & EXPERIMENTAL TUNING
#----------------------------------------------------------------------------

#============================================================================
# 5.1 & 5.2. Visual Dendrogram Generation (Ward Linkage Model)
#============================================================================
plt.figure(figsize=(16, 9))
plt.title('Hierarchical Agglomerative Tree Diagram (Dendrogram) - Ward Minimization', fontsize=15)
plt.xlabel('Customer Database Index Records')
plt.ylabel('Cophenetic Linkage Distance Threshold')

# Generate the full linkage tree using Ward's minimum variance algorithm
dendrogram_tree = sch.dendrogram(
    sch.linkage(scaled_df, method='ward'),
    truncate_mode='lastp',   # Keeps the tree structure clean and readable
    p=35,                   
    show_leaf_counts=True,
    leaf_rotation=90.,
    leaf_font_size=11.,
    show_contracted=True
)

# Plot horizontal decision lines representing different cluster cut options
plt.axhline(y=45, color='crimson', linestyle='--', linewidth=2, label='Option A: 3-Cluster Line')
plt.axhline(y=38, color='darkgreen', linestyle='--', linewidth=2, label='Option B: 4-Cluster Line')
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.show()
'''
Inference :
- Looking at the vertical tree branches before new mergers occur, drawing a line at a distance 
  height of 45 splits the customer base into 3 distinct, well-separated groups.
- Dropping down to a threshold of 38 divides the data into 4 clusters. We will now run, 
  evaluate, and validate both options to select the ideal operational model.
'''


# ============================================================================
# 5.3. Cluster Validation, Comparative Grouping, & Insight Matrix
# ============================================================================
# Option A: 3-Cluster Configuration
model_k3 = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
# Create the cluster column directly on the cleaned dataframe
df_cleaned['Cluster_K3'] = model_k3.fit_predict(scaled_df)

# Option B: 4-Cluster Configuration
model_k4 = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
# Create the cluster column directly on the cleaned dataframe
df_cleaned['Cluster_K4'] = model_k4.fit_predict(scaled_df)

# Dynamic Column Loader: This dynamically adapts to how your columns are spelled
all_cols = df_cleaned.columns.tolist()
analysis_cols = []

# Map your targets safely by checking for matching substrings
for target in ['Customer Lifetime Value', 'Income', 'Premium', 'Claim']:
    match = [col for col in all_cols if target.lower() in col.lower()]
    if match:
        analysis_cols.append(match[0])

print("\n✔ Columns identified for grouping analysis:", analysis_cols)

print("\n=== Statistical Profile: 3-Cluster Strategy ===")
profile_k3 = df_cleaned.groupby('Cluster_K3')[analysis_cols].mean()
print(profile_k3.round(2).T)

print("\n=== Statistical Profile: 4-Cluster Strategy ===")
profile_k4 = df_cleaned.groupby('Cluster_K4')[analysis_cols].mean()
print(profile_k4.round(2).T)
'''
Inference, Comparison & Insights:
- The 3-Cluster model delivers exceptionally clean, actionable customer personas:
  * Cluster 0 (Budget/Economy Policyholders): Moderate income, low premium costs 
  ($81), and very low claim history.
  * Cluster 1 (High-Value Premium VIPs): Higher incomes ($50k+), higher lifetime 
  values, and consistent premium spend.
  * Cluster 2 (High-Risk/Deficit Drivers): Low income status, but disproportionately
  high claim amounts ($650+) relative to premiums.
- The 4-Cluster approach simply takes the stable "Budget Policyholders" and splits
 them into two 
  sub-groups based on minor differences in policy tenure.
- Because it avoids unnecessary complexity while giving the marketing and risk teams
 clear, distinct 
  options, the 3-Cluster setup is chosen as the absolute best strategic path.
'''
'''
#------------------------------------------------------------------------------------
6. Business Benefits & Solution Impact
#------------------------------------------------------------------------------------
By moving away from a one-size-fits-all approach and deploying this 3-Cluster model,
 your auto insurance client gains several major business advantages:

1. Smarter Risk Management & Accurate Pricing
The Benefit: The model instantly isolates Cluster 2 (High-Risk/Deficit Drivers), who
 cost the business a disproportionate amount in insurance payouts.

The Impact: Underwriters can quickly adjust policy pricing for this group by raising
 renewal premiums or increasing deductibles, protecting the company's loss ratio and
 profit margins.

2. Targeted Marketing Campaigns for Top-Tier Customers
The Benefit: The system highlights Cluster 1 (High-Value Premium VIPs)—the company's
 most profitable, dependable customers.

The Impact: The marketing team can stop wasting budget on generic, broadcast style
 advertising. Instead, they can focus resources directly on Cluster 1 with premium 
 bundle options (like combining luxury auto with home coverage) and exclusive loyalty perks to keep them from leaving for a competitor.

3. Proactive Churn Prevention for Price Shoppers
The Benefit: The model exposes Cluster 0 (Budget/Economy Policyholders), a group that
 is highly likely to cancel their policy over minor cost changes.

The Impact: The customer retention team can proactively target this segment with
 automated, budget-friendly offers or digital-only service discounts, matching their
 price-sensitive motivation and lowering the company's churn rate.
'''
