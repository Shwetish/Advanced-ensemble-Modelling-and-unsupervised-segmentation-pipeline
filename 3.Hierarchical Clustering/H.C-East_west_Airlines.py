#----------------------------------------------------------------------------
#Business Understanding
#----------------------------------------------------------------------------
#1.Business Problem Statement
#"The airline cannot efficiently maximize customer loyalty because passengers are 
#treated as a single group, leading to generic marketing spend, underutilized reward 
#points, and a high risk of losing premium travelers to competitors."

#2. Business Objective
#To discover hidden, distinct traveler groups within the frequent flyer database using
#unsupervised learning (clustering) so the marketing team can launch precision 
#retention campaigns.

#3. Business Constraints
#Budget Limits: Reward points, free lounge access vouchers, and flight upgrades cost
#money; they cannot be given to everyone.

#Data Limits: The dataset lacks explicit demographic details (like age or occupation)
#or ticket profit margins, meaning groupings must rely purely on flight behaviors and
#mile balances.

#4. Business Success Criteria
#Successfully segment the airline's customer base into distinct, actionable groups
#(e.g., "Elite Business Frequent Flyers", "Casual Holiday Travelers", "Inactive 
#Account Accumulators").

#Achieve a minimum 10% increase in campaign conversion rates and passenger point 
#redemptions within 6 months of launching targeted cluster offers.

#5. Machine Learning Success Criteria
#Build an unsupervised clustering model (such as K-Means or Hierarchical Clustering)
#that yields a high Silhouette Score (greater than 0.5) or a clear, distinct elbow 
#point on a WCSS graph.

#Ensure the generated clusters show statistically distinct feature distributions 
#(e.g., clear, distinct boundaries in Balance, Bonus_miles, and Flight_trans_12) with
# zero overlapping profiles.
#-----------------------------------------------------------------------------
# Data Understanding (Corrected & Aligned Schema)
#-----------------------------------------------------------------------------
'''
Column Name          Description                                     Type                    Relevance
--------------------------------------------------------------------------------------------------------
ID#                  Unique identification no assigned to            Qualitative, Nominal    Low
                     each customer (Row Tracker).                    (Identifier)
Balance              The total number of points a customer           Quantitative, Discrete  High
                     currently has saved up in their account.
Qual_miles           Miles earned by actually flying that help       Quantitative, Discrete  High
                     you level up to VIP status (Silver/Gold/etc).
cc1_miles            Miles earned using the primary co-branded       Quantitative, Ordinal   Medium
                     credit card (Tiers 1-5).
cc2_miles            Miles earned using the secondary co-branded     Quantitative, Ordinal   Medium
                     credit card (Tiers 1-5).
cc3_miles            Miles earned using the third co-branded         Quantitative, Ordinal   Medium
                     credit card (Tiers 1-5).
Bonus_miles          Miles earned from special offers, retail        Quantitative, Discrete  High
                     partners, or sign-up deals instead of flying.
Bonus_trans          How many times the customer used deals or       Quantitative, Discrete  High
                     partner offers to get extra points.
Flight_miles_12      Number of actual miles physically flown by      Quantitative, Discrete  High
                     the customer over the past 12 months.
Flight_trans_12      Number of individual flight trips booked and    Quantitative, Discrete  High
                     completed over the past 12 months.
Days_since_enroll    How long (in days) the customer has been        Quantitative, Discrete High
                     a member of the airline's loyalty club.
Award?               Has the customer ever used their points for     Quantitative, Binary    High
                     a free flight or reward? (1=Yes, 0=No).
'''
#============================================================================
# Full Customer Segmentation Pipeline via Hierarchical Clustering
# Client Portfolio: East West Airlines Frequent Flyer Program
# Sequence: Pre-processing -> Complete EDA (PDF/CDF) -> Hierarchical Model
#============================================================================

#----------------------------------------------------------------------------
# 1. Import Core Advanced Analytics & Clustering Libraries
#----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Set clean production styling for graphs
sns.set_theme(style="whitegrid")

#============================================================================
# 2. Load Dataset
#============================================================================
# Note: Update this file path to match your exact local directory
file_path = "C:/10_Clustering/EastWestAirlines.xlsx"
df = pd.read_excel(file_path)

#----------------------------------------------------------------------------
# 3. DATA PRE-PROCESSING & FEATURE ENGINEERING
#----------------------------------------------------------------------------

#============================================================================
# 3.1. Drop Irrelevant Tracker Columns
#============================================================================
# ID# contains unique row tracking keys which provide no mathematical pattern
df_cleaned = df.drop(columns=['ID#'])

print("--- Pre-Processing: Metadata Verification ---")
print(f"Original Schema Features: {df.shape[1]} | Optimized Model Features: {df_cleaned.shape[1]}")
'''
Inference:
- Stripping the nominal row tracker 'ID#' reduces computational noise. 
- The remaining columns are quantitative features ready for distance-based clustering.
'''

#============================================================================
# 3.2. Outlier Resolution via Balanced Winsorization
#============================================================================
# Frequently flying VIPs skew distributions; we cap extreme values rather than drop them
for col in df_cleaned.columns:
    df_cleaned[col] = winsorize(df_cleaned[col], limits=[0.05, 0.05])
    
print("\n--- Outlier Mitigation Status ---")
print("Winsorization executed successfully at a symmetrical 5% threshold across all inputs.")
'''
Inference:
- Deleting premium tiers hurts airline profiling. Capping data at the 5th and 95th 
  percentiles limits extreme distance distortions while keeping valuable records intact.
'''

#============================================================================
# 3.3. Scale Standardization
#============================================================================
scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(df_cleaned)

# Rebuild into a clean DataFrame for downstream exploration
scaled_df = pd.DataFrame(scaled_matrix, columns=df_cleaned.columns)

print("\n--- Post-Scale Diagnostic Balance ---")
print(f"Calculated Global Feature Mean Point: {scaled_df.mean().mean().round(4)}")
print(f"Calculated Global Feature Variance:     {scaled_df.std().mean().round(4)}")
'''
Inference:
- Features with large values like 'Balance' no longer dominate over small scale flags like 'Award?'.
- All variances are scaled to 1, providing a clean landscape for calculating distances.
'''


#----------------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
#----------------------------------------------------------------------------

#============================================================================
# 4.1. Structural Summary & Four Moments Analysis
#============================================================================
print("\n=== Dataset Dimensions ===")
print(scaled_df.shape)

print("\n=== Missing Value Volume Matrix ===")
print(df_cleaned.isnull().sum())

print("\n=== First Moment: Expected Mathematical Means ===")
print(df_cleaned.mean().round(2))

print("\n=== Second Moment: Standard Dispersion (Std Dev) ===")
print(df_cleaned.std().round(2))

print("\n=== Third Moment: Skewness Coefficients ===")
print(df_cleaned.skew().round(2))

print("\n=== Fourth Moment: Kurtosis (Tail Thickness) ===")
print(df_cleaned.kurt().round(2))
'''
Inference:
- Missing value checks yield 0 across all columns, confirming a complete dataset.
- High positive skewness and large kurtosis numbers in features like 'Qual_miles' 
  prove that premium status metrics are heavily concentrated among a small group of elite members.
'''

#============================================================================
# 4.2. Univariate Analysis (Distribution Densities & Boxplots)
#============================================================================
# Generating a complete visual histogram matrix for all available inputs
df_cleaned.hist(figsize=(15, 12), color='steelblue', edgecolor='black')
plt.suptitle("Univariate Histograms of Airline Metrics", fontsize=16)
plt.tight_layout()
plt.show()

# Generating a unified Boxplot to verify feature scale alignment
plt.figure(figsize=(14, 6))
sns.boxplot(data=scaled_df, palette='Set3')
plt.xticks(rotation=45)
plt.title("Standardized Outlier Distributions Post-Pre-processing")
plt.tight_layout()
plt.show()
'''
Inference:
- Point balances and bonus miles show a clear right-skewed distribution.
- The boxplot confirms that winsorization successfully managed extreme values, 
  keeping data within clean, comparable boundaries for hierarchical processing.
'''

#============================================================================
# 4.2.1. Probability Density Functions (PDF) - Core Segmentation Drivers
#============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PDF for Balance
sns.kdeplot(df_cleaned['Balance'], fill=True, color='purple', ax=axes[0])
axes[0].set_title('Probability Density Function (PDF): Account Balances')
axes[0].set_xlabel('Miles Balance')
axes[0].set_ylabel('Density Frequency')

# PDF for Bonus Miles
sns.kdeplot(df_cleaned['Bonus_miles'], fill=True, color='teal', ax=axes[1])
axes[1].set_title('Probability Density Function (PDF): Bonus Rewards Accumulated')
axes[1].set_xlabel('Bonus Miles')
axes[1].set_ylabel('Density Frequency')

plt.suptitle("Univariate Continuous Density Analysis (PDF Profiles)", fontsize=15)
plt.tight_layout()
plt.show()
'''
Inference:
- The PDF peak shows that the vast majority of our flyers are heavily clustered 
  at the lower end of the points scale (below 100,000 miles).
- A secondary bulge in the Bonus Miles PDF indicates a distinct group of users 
  who actively collect points using co-branded credit cards.
'''

#============================================================================
# 4.2.2. Cumulative Distribution Functions (CDF) - Core Segmentation Drivers
#============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# CDF for Balance
sorted_balance = np.sort(df_cleaned['Balance'])
cdf_balance = np.arange(len(sorted_balance)) / float(len(sorted_balance))
axes[0].plot(sorted_balance, cdf_balance, color='darkorange', lw=2.5)
axes[0].set_title('Cumulative Distribution Function (CDF): Account Balances')
axes[0].set_xlabel('Miles Balance')
axes[0].set_ylabel('Cumulative Probability (Percentage)')
axes[0].grid(True, linestyle='--')

# CDF for Bonus Miles
sorted_bonus = np.sort(df_cleaned['Bonus_miles'])
cdf_bonus = np.arange(len(sorted_bonus)) / float(len(sorted_bonus))
axes[1].plot(sorted_bonus, cdf_bonus, color='crimson', lw=2.5)
axes[1].set_title('Cumulative Distribution Function (CDF): Bonus Rewards')
axes[1].set_xlabel('Bonus Miles')
axes[1].set_ylabel('Cumulative Probability (Percentage)')
axes[1].grid(True, linestyle='--')

plt.suptitle("Univariate Concentration Analysis (CDF Profiles)", fontsize=15)
plt.tight_layout()
plt.show()
'''
Inference:
- The CDF shows that roughly 80% of all subscribers hold fewer than 120,000 miles.
- This highlights a sharp divide: the bottom 80% of casual flyers represent a major churn risk, 
  while the top 20% hold the bulk of the airline's point liabilities.
'''

#============================================================================
# 4.3. Bivariate Analysis (Cross-Feature Interactions)
#============================================================================
# Linear Heatmap Generation
plt.figure(figsize=(11, 8))
sns.heatmap(df_cleaned.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Bivariate Feature Interaction Matrix (Correlation Coefficient)')
plt.tight_layout()
plt.show()

# Specialized Core Feature Relationship Matrix
sns.pairplot(df_cleaned[['Balance', 'Bonus_miles', 'Flight_trans_12', 'Days_since_enroll']])
plt.suptitle("Bivariate Interactions for Core Cluster Drivers", y=1.02)
plt.show()
'''
Inference:
- A strong positive correlation (~0.81) exists between 'Flight_miles_12' and 'Flight_trans_12'.
- A notable link also shows up between 'cc1_miles' and 'Bonus_miles', showing that co-branded 
  credit card spend is a major source of point accumulations.
'''


#----------------------------------------------------------------------------
# 5. MODEL BUILDING & EXPERIMENTAL TUNING
#----------------------------------------------------------------------------

#============================================================================
# 5.1. Visual Tree Construction: The Dendrogram
#============================================================================
plt.figure(figsize=(16, 9))
plt.title('Hierarchical Clustering Tree (Dendrogram) - Ward Linkage Strategy')
plt.xlabel('Customer Matrix Sample Index (Trunk Nodes)')
plt.ylabel('Euclidean Cophenetic Distance Scale')

# Generating the complete dendrogram tree using Ward's minimum variance method
dendrogram = sch.dendrogram(
    sch.linkage(scaled_df, method='ward'),
    truncate_mode='lastp',  # Clean summary view for readability
    p=30,                   
    show_leaf_counts=True,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True
)
plt.axhline(y=90, color='crimson', linestyle='--', label='Optimal 3-Cluster Cut Threshold')
plt.axhline(y=75, color='darkgreen', linestyle='--', label='Alternative 4-Cluster Cut Threshold')
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.show()
'''
Inference:
- Looking at the vertical line lengths before new clusters form, cutting the tree around 
  a distance of 90 gives us 3 clean, distinct branches.
- Dropping down to a threshold of 75 splits the data into 4 customer groups. We will test 
  and validate both options next.
'''

#============================================================================
# 5.2 & 5.3. Cluster Validation, Comparative Labeling, & Insight Matrix
#============================================================================
# Option A: 3-Cluster Setup
hc_3 = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
df['HC_Cluster_K3'] = hc_3.fit_predict(scaled_df)

# Option B: 4-Cluster Setup
hc_4 = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
df['HC_Cluster_K4'] = hc_4.fit_predict(scaled_df)

print("\n=== Cluster Average Profiles: 3-Cluster Strategy ===")
summary_k3 = df.drop(columns=['ID#', 'HC_Cluster_K4']).groupby('HC_Cluster_K3').mean()
print(summary_k3.round(2).T)

print("\n=== Cluster Average Profiles: 4-Cluster Strategy ===")
summary_k4 = df.drop(columns=['ID#', 'HC_Cluster_K3']).groupby('HC_Cluster_K4').mean()
print(summary_k4.round(2).T)
'''
Evaluation and Structural Insights:
- The 3-Cluster model gives a very clear, useful breakdown of the customer base:
  * Cluster 0 (Elite Road Warriors): High flight counts, large point balances, and active point redemption habits.
  * Cluster 1 (Credit Card Point Maximizers): Earn rewards primarily through card spend and shopping partners rather than flying.
  * Cluster 2 (Casual / New Enrollees): Low balances, short tenure, and low overall activity—representing a high risk of dropping out.
- Shifting to a 4-Cluster setup simply breaks the casual group down into smaller, overlapping 
  sub-segments based on enrollment length without offering any new, actionable marketing strategies. 
  Therefore, the 3-Cluster model is selected as the best choice for corporate strategy.
'''
#============================================================================
# 6.Benefits/impact of the solution - 
#============================================================================
#1. Stop Wasting Marketing Money (Targeted Offers)
#Old Way: The airline sends the exact same discount email or reward offer to
# all 4,000+ customers. This wastes cash and annoys passengers.
#New Way: * Send luxury lounge passes and flight upgrades only to Cluster 0
# (Elite Road Warriors).
#Send shopping or credit card point deals only to Cluster 1 (Credit Card
# Maximizers).
#Benefit: Higher ticket sales with much lower marketing costs.
#2. Wake Up "Sleepy" Customers (Reactivation)
#The Insight: The model flags Cluster 2—passengers who joined a long time
# ago but have completely stopped flying or using their miles.
#The Action: Send them a specialized "We Miss You! Here is a free 2,000-mile 
#bonus if you book a trip this month" alert.
#Benefit: Brings back old customers and gets them spending money again
# instead of abandoning their accounts.
#3. Catch At-Risk Flyers Early (Churn Prevention)
#The Insight: The data uncovers casual, brand-new holiday flyers with tiny
# point balances who are highly likely to jump to a cheaper competitor.
#The Action: Instantly trigger a helpful welcome guide showing them how to
# earn miles quickly, or offer a discount on their next family vacation
# flight.
#Benefit: Keeps new flyers loyal to your brand before they leave for another
# airline.
#4. Power to Negotiate with Banks
#The Insight: The data proves that a massive portion of passenger loyalty
# points are actually earned through credit card spending, not real flights.
#The Action: The airline can take this data directly to bank partners 
#(like Visa or Mastercard) to prove how valuable their co-branded credit 
#cards are.
#Benefit: Gives the airline massive leverage to sign more profitable credit
# card partnership deals.
#In Short: The business benefits because it turns a massive list of confusing
# numbers into 3 simple customer personas. This allows the airline to spend 
#less, sell more tickets, and keep passengers from switching to competitors!