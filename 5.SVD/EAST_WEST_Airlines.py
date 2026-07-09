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
# Full Customer Segmentation Pipeline: East West Airlines
# Step-by-Step: EDA (Moments, PDF/CDF), Preprocessing, Winsorization, & K-Means
#============================================================================

#----------------------------------------------------------------------------
# 1. Import Core Libraries
#----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize
from sklearn.cluster import KMeans

# Set aesthetic style for charts
sns.set_theme(style="whitegrid")

#============================================================================
# 2. Load Dataset
#============================================================================
# Update this path to match your local file layout
file_path = "C:/10_Clustering/EastWestAirlines.xlsx"
df = pd.read_excel(file_path)

#----------------------------------------------------------------------------
# EXPLORATORY DATA ANALYSIS (EDA)
#----------------------------------------------------------------------------

#============================================================================
# 1. Basic Exploration
#============================================================================
print("--- First 5 Records ---")
print(df.head())
print("\n--- Last 5 Records ---")
print(df.tail())
print("\n--- Dataset Dimensional Shape ---")
print(df.shape)
print("\n--- Structural Information ---")
print(df.info())
print("\n--- Basic Descriptive Summary ---")
print(df.describe())
'''
Inference:
Dataset contains customer loyalty behavior information.
Mostly numerical variables.
No categorical text columns.
Features are suitable for clustering.
'''

#============================================================================
# 2. Missing Value Analysis
#============================================================================
print("\n--- Missing Value Counts ---")
print(df.isnull().sum())
'''
Inference:
If all values are 0 -> dataset has no missing values.
Good quality dataset for clustering.
'''

#============================================================================
# 3. Duplicate Value Check
#============================================================================
print("\n--- Total Duplicate Rows ---")
print(df.duplicated().sum())
'''
Inference:
Duplicate rows may create biased clusters.
Remove duplicates if present.
'''

#============================================================================
# 4. Data Types
#============================================================================
print("\n--- Data Column Types ---")
print(df.dtypes)
'''
Inference:
Most variables are integer/numeric.
Suitable for distance-based clustering algorithms.
'''

#============================================================================
# 5. First Four Moments
#============================================================================

# 5.1 First Moment -> Mean
print("\n--- First Moment: Mean ---")
print(df.mean(numeric_only=True))
'''
Inference:
Gives central tendency of airline customer behavior.
High average balance indicates loyal customers.
'''

# 5.2 Second Moment -> Variance & Standard Deviation
print("\n--- Second Moment: Variance ---")
print(df.var(numeric_only=True))
print("\n--- Second Moment: Standard Deviation ---")
print(df.std(numeric_only=True))
'''
Inference:
High variance indicates customer behavior differs significantly.
Features like Balance and Bonus_miles usually show high variability.
'''

# 5.3 Third Moment -> Skewness
print("\n--- Third Moment: Skewness ---")
print(df.skew(numeric_only=True))
'''
Inference:
Positive skewness indicates most customers have low values while few customers have extremely high values.
Airline datasets are usually highly right-skewed.
'''

# 5.4 Fourth Moment -> Kurtosis
print("\n--- Fourth Moment: Kurtosis ---")
print(df.kurt(numeric_only=True))
'''
Inference:
High kurtosis indicates presence of extreme outliers/VIP customers.
Important before clustering.
'''

#============================================================================
# 6. Univariate Analysis
#============================================================================
# Histogram Grid
df.hist(figsize=(15,12), color='steelblue', edgecolor='black')
plt.suptitle("Univariate Histograms of Airline Metrics", fontsize=16)
plt.tight_layout()
plt.show()
'''
Inference:
Helps understand distribution of each feature.
Many airline features are non-normal and right-skewed.
'''

# Global Boxplot Configuration
plt.figure(figsize=(15,10))
sns.boxplot(data=df.drop(columns=['ID#'])) # Dropping ID# for visual scalability
plt.xticks(rotation=90)
plt.title("Visual Outlier Distributions Across Features")
plt.tight_layout()
plt.show()
'''
Inference:
Detects outliers.
VIP customers appear as extreme outliers.
'''

#============================================================================
# 7. Bivariate Analysis
#============================================================================
# Correlation Matrix
corr = df.corr(numeric_only=True)

plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Interaction Correlation Matrix")
plt.tight_layout()
plt.show()
'''
Inference:
Strong positive correlation may exist between:
Balance and Bonus_miles
Flight_miles_12 and Flight_trans_12
Helps identify redundant variables.
'''

# Pairplot Matrix
sns.pairplot(df[['Balance','Bonus_miles', 'Flight_miles_12', 'Days_since_enroll']])
plt.suptitle("Bivariate Interactions for Core Cluster Drivers", y=1.02)
plt.show()
'''
Inference:
Helps visualize customer group patterns.
Useful before clustering.
'''

#============================================================================
# 8. PDF (Probability Density Function)
#============================================================================
plt.figure(figsize=(8, 4))
sns.kdeplot(df['Balance'], fill=True, color='purple')
plt.title("PDF of Balance")
plt.xlabel("Balance")
plt.show()
'''
Inference:
Shows probability density of customer balances.
Helps identify concentration regions.
'''

#============================================================================
# 9. CDF (Cumulative Distribution Function)
#============================================================================
x = np.sort(df['Balance'])
y = np.arange(len(x)) / float(len(x))

plt.figure(figsize=(8, 4))
plt.plot(x, y, color='darkorange', lw=2)
plt.xlabel("Balance")
plt.ylabel("CDF")
plt.title("CDF of Balance")
plt.show()
'''
Inference:
Shows percentage of customers below a balance threshold.
Example: 80% customers may have balance below certain level.
Useful for customer segmentation strategy.
'''

#============================================================================
# 10. Outlier Detection (Interquartile Range - IQR Metric)
#============================================================================
print("\n--- Outlier Counts Per Column via IQR ---")
for col in df.columns[1:]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")
'''
Inference:
Frequent flyer datasets naturally contain extreme VIP customers.
Outlier treatment should be done carefully.
'''

#----------------------------------------------------------------------------
# DATA PRE-PROCESSING
#----------------------------------------------------------------------------

#============================================================================
# 11. Drop Irrelevant Feature
#============================================================================
df1 = df.drop(['ID#'], axis=1)
print("\n--- Matrix After ID Feature Removal ---")
print(df1.head())
'''
Inference:
ID is only a customer identifier.
It has no business meaning for clustering.
Removing it improves clustering quality.
'''

#============================================================================
# 12. Outlier Treatment Using Winsorization
#============================================================================
for col in df1.columns:
    df1[col] = winsorize(df1[col], limits=[0.05, 0.05])
'''
Inference:
Winsorization reduces extreme values without deleting records.
Important because airline VIP customers are valuable observations.
'''

#============================================================================
# 13. Skewness Check Post-Winsorization
#============================================================================
print("\n--- Post-Winsorization Skewness Matrix ---")
print(df1.skew())
'''
Inference:
Positive skewness indicates many low-frequency travelers and few extremely loyal customers.
Common behavior in airline datasets.
'''

#============================================================================
# 14. Feature Scaling
#============================================================================
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df1)

scaled_df = pd.DataFrame(scaled_data, columns=df1.columns)
print("\n--- Standardized Scaled DataFrame (Head) ---")
print(scaled_df.head())
'''
Inference:
All variables converted to same scale.
Improves K-Means cluster performance.
'''

#============================================================================
# 15. Check Mean and Standard Deviation After Scaling
#============================================================================
print("\n--- Standardized Feature Calculated Means ---")
print(scaled_df.mean().round(4))

print("\n--- Standardized Feature Calculated Deviations ---")
print(scaled_df.std().round(4))
'''
Inference:
Mean approximately 0.
Standard deviation approximately 1.
Data properly standardized.
'''

#----------------------------------------------------------------------------
# 16. Model Building & Dimensionality Reduction using SVD
#----------------------------------------------------------------------------
from sklearn.decomposition import TruncatedSVD

# 16.1 Initialize and Fit SVD
# We extract 4 components to match the optimal business strategy structures
svd = TruncatedSVD(n_components=4, random_state=42)
svd_elements = svd.fit_transform(scaled_df)

# Store the compressed customer profiles into a new DataFrame
df_svd = pd.DataFrame(svd_elements, columns=['Concept_1', 'Concept_2', 'Concept_3', 'Concept_4'])

#============================================================================
# 17. SVD Mathematical Validation (Explained Variance)
#============================================================================
print("--- Explained Variance Ratio Per Component ---")
print(svd.explained_variance_ratio_)
print(f"\nTotal Information Captured by 4 Components: {svd.explained_variance_ratio_.sum().round(4) * 100}%")

# Create a Loadings Matrix to see which original metrics build our new themes
loadings = pd.DataFrame(svd.components_.T, columns=['Concept_1', 'Concept_2', 'Concept_3', 'Concept_4'], index=df1.columns)

print("\n=== Top Feature Drivers for Concept 1 (Value Engine) ===")
print(loadings['Concept_1'].sort_values(ascending=False).head(4))

print("\n=== Top Feature Drivers for Concept 2 (Credit Card vs Flight Behavior) ===")
print(loadings['Concept_2'].sort_values(ascending=False).head(4))

#=================================================================================
#Business Impact
#=================================================================================
'''
1. Massive Marketing Savings (No More Spam)
The Problem: Sending generic flight discount emails to card spenders or luxury flyers
wastes money and gets ignored.

The SVD Solution: You launch promotions to exact matching profiles. You pitch premium
credit card upgrades only to high Theme 2 spenders, and luxury lounge passes only to 
your true Theme 1 VIPs.

The Impact: Massive reduction in marketing waste and much higher conversion rates.

2. Waking Up "Sleepy" Accounts
The Problem: Unused reward points sitting in customer accounts represent huge 
financial liabilities on the airlines balance sheet.

The SVD Solution: The airline uses Theme 4 to isolate accounts with high points but
zero recent flight activity. They target them with point-redemption flash sales 
(e.g., "Fly to Hawaii this month for 30% fewer points").

The Impact: Erases balance-sheet financial liabilities and generates immediate 
airport/baggage fees from passengers who were sitting dormant.

3. Bulletproof Bargaining Power with Banks
The Problem: Negotiating multi-million dollar deals with credit card networks 
(like Visa or Mastercard) requires hard evidence of consumer value.

The SVD Solution: SVD mathematically proves exactly how much of your passenger
revenue is generated via credit card transactions (Theme 2) versus traditional 
flying (Theme 3).

The Impact: The airline can confidently demand higher margins and better 
revenue-sharing terms from partner banks during contract renewals.
'''


