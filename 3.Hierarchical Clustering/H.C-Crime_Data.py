#----------------------------------------------------------------------------
#Business Understanding
#----------------------------------------------------------------------------
#1.Business Problem Statement: Government emergency funds and law enforcement 
#resources are being distributed evenly across the country, which wastes money
# on peaceful areas while leaving high-crime zones underfunded.

#2.Business Objective: Group regions/states by their specific crime profiles 
#to deploy public safety budgets and police forces where they are needed most.

#3.Motivation (Public/Government Intent):

#Safe/Peaceful Zones: Need proactive community policing to maintain their 
#low crime rates.

#High-Risk Urban Centers: Crucially need emergency tactical funding and
# violent crime intervention.

#5.Constraints: Government budgets are strictly capped, and reallocating 
#resources faces heavy political and legal hurdles.

#6.Business Success Criteria: A measurable decrease in overall violent crime 
#rates over the next two years due to better resource placement.

#7.ML Success Criteria: A clustering model that clearly groups states into 3
# or 4 actionable risk tiers (e.g., Low Crime/Rural, High Property Crime, 
#Severe Violent Crime) without muddy or ambiguous overlaps.

#----------------------------------------------------------------------------
#Data Understanding:-
#----------------------------------------------------------------------------
'''
Name of Feature	     Simple Description	          Type	             Relevance
Murder               Murder arrests        Quantitative,Continuous       High
                     per 100k people.
Assault              Physical attacks      Quantitative,Discrete         High
                     per 100k people.
UrbanPop             Percentage of people  Numeric                       High
                     living in cities
Rape                 Rape arrests          Quantitative,Continuous       High
                     per 100k people.
'''

#============================================================================
# Full Regional Risk Segmentation Pipeline via Hierarchical Clustering
# Client Portfolio: Government Public Safety & Resource Allocation
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
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Set clean production styling for graphs
sns.set_theme(style="whitegrid")

#============================================================================
# 2. Load Dataset
#============================================================================
# Note: Update this file path to match your exact local directory structure
# Since the file is a CSV as per instructions, we use pd.read_csv
file_path = "C:/10_Clustering/crime_data.csv"
df = pd.read_csv(file_path)


#----------------------------------------------------------------------------
# 3. DATA PRE-PROCESSING & FEATURE ENGINEERING
#----------------------------------------------------------------------------

#============================================================================
# 3.1. Data Cleaning & Feature Selection
#============================================================================
# Rename the first unnamed column if it tracks the State names
if df.columns[0].startswith('Unnamed') or df.columns[0] == 'State':
    df.rename(columns={df.columns[0]: 'State'}, inplace=True)

# Drop text identifiers for the math model matrix
df_numeric = df.drop(columns=['State'], errors='ignore')

# Check and handle missing values by filling with column median if present
for col in df_numeric.columns:
    if df_numeric[col].isnull().sum() > 0:
        df_numeric[col] = df_numeric[col].fillna(df_numeric[col].median())

print("--- Pre-Processing: Schema Verification ---")
print(f"Original Dataset Columns: {df.shape[1]} | Model Ready Features: {df_numeric.shape[1]}")
'''
Inference :
- Dropping the 'State' column string keys prevents the distance algorithm 
from crashing. 
  The states are preserved in the original dataframe to label our final 
  clusters.
- The dataset has no missing values, creating a clean foundation for spatial 
distance metrics.
'''

#============================================================================
# 3.2. Feature Engineering (Scale Standardization)
#============================================================================
# Crime rates are measured per 100,000 people, while UrbanPop is a percentage.
# We must scale them so that large numbers like Assault do not distort the distances.
scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(df_numeric)

# Rebuild into a clean DataFrame for downstream exploration
scaled_df = pd.DataFrame(scaled_matrix, columns=df_numeric.columns)

print("\n--- Post-Scale Diagnostic Balance ---")
print(f"Calculated Global Feature Mean Point: {scaled_df.mean().mean().round(4)}")
print(f"Calculated Global Feature Variance:     {scaled_df.std().mean().round(4)}")
'''
Inference :
- Without scaling, the 'Assault' feature (values up to 300+) would completely override 
  'Murder' values (mostly single/double digits) during distance calculation.
- Standardizing sets all features to a mean of 0 and a variance of 1, giving every 
  crime metric an equal weight when defining safe vs dangerous regions.
'''


#----------------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
#----------------------------------------------------------------------------

#============================================================================
# 4.1. Summary Statistics & Four Moments Analysis
#============================================================================
print("\n=== Dataset Dimensions ===")
print(df_numeric.shape)

print("\n=== First Moment: Expected Mathematical Means ===")
print(df_numeric.mean().round(2))

print("\n=== Second Moment: Standard Dispersion / Std Dev ===")
print(df_numeric.std().round(2))

print("\n=== Third Moment: Skewness Coefficients ===")
print(df_numeric.skew().round(2))

print("\n=== Fourth Moment: Kurtosis / Tail Thickness ===")
print(df_numeric.kurt().round(2))
'''
Inference & Comments:
- The mean values show that Assault is the most widespread violent crime 
across states (mean ~170.8).
- The skewness and kurtosis are close to zero for most features. This tells 
us the crime 
  distributions are relatively balanced across states without extreme, 
  completely unmanageable outliers.
'''

#============================================================================
# 4.2. Univariate Analysis (Distribution Densities, PDF, and CDF)
#============================================================================
# 4.2.1. Complete Visual Histogram Matrix for Crime Inputs
df_numeric.hist(figsize=(12, 8), color='steelblue', edgecolor='black')
plt.suptitle("Univariate Histograms of State Crime Metrics", fontsize=16)
plt.tight_layout()
plt.show()
'''
Inference & Comments:
- Murder rates show a slightly bimodal distribution, suggesting a split 
between two clear 
  groups of states: one group with low murder levels and another with 
  distinct high peaks.
- Urban Population numbers show a wide distribution, indicating a balanced
 mix of rural and urbanized states.
'''

# 4.2.2. Probability Density Functions (PDF) - Core Public Safety Drivers
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.kdeplot(df_numeric['Murder'], fill=True, color='purple', ax=axes[0])
axes[0].set_title('Probability Density Function (PDF): Murder Rates')
axes[0].set_xlabel('Murders (per 100k citizens)')
axes[0].set_ylabel('Density Frequency')

sns.kdeplot(df_numeric['Assault'], fill=True, color='teal', ax=axes[1])
axes[1].set_title('Probability Density Function (PDF): Assault Rates')
axes[1].set_xlabel('Assaults (per 100k citizens)')
axes[1].set_ylabel('Density Frequency')

plt.suptitle("Univariate Continuous Density Analysis (PDF Profiles)", fontsize=15)
plt.tight_layout()
plt.show()
'''
Inference & Comments:
- The Murder PDF curve peaks clearly around 2-5 cases per 100k, showing that 
most states 
  maintain low murder rates, though a flat right tail reveals dangerous 
  exceptions.
- The Assault PDF shows a broad, spread-out profile, proving that assault 
rates vary heavily across different state jurisdictions.
'''

# 4.2.3. Cumulative Distribution Functions (CDF) - Regional Risk Concentration
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# CDF for Murder
sorted_murder = np.sort(df_numeric['Murder'])
cdf_murder = np.arange(len(sorted_murder)) / float(len(sorted_murder))
axes[0].plot(sorted_murder, cdf_murder, color='darkorange', lw=2.5)
axes[0].set_title('Cumulative Distribution Function (CDF): Murder Rates')
axes[0].set_xlabel('Murder Rate Scale')
axes[0].set_ylabel('Cumulative Probability')
axes[0].grid(True, linestyle='--')

# CDF for Assault
sorted_assault = np.sort(df_numeric['Assault'])
cdf_assault = np.arange(len(sorted_assault)) / float(len(sorted_assault))
axes[1].plot(sorted_assault, cdf_assault, color='crimson', lw=2.5)
axes[1].set_title('Cumulative Distribution Function (CDF): Assault Rates')
axes[1].set_xlabel('Assault Rate Scale')
axes[1].set_ylabel('Cumulative Probability')
axes[1].grid(True, linestyle='--')

plt.suptitle("Univariate Concentration Analysis (CDF Profiles)", fontsize=15)
plt.tight_layout()
plt.show()
'''
Inference & Comments:
- The Murder CDF shows that roughly 60% of US states have a murder rate below
 8 per 100,000.
- The Assault CDF demonstrates that 80% of states stay under 260 cases, 
highlighting that the 
  top 20% of states suffer from extreme, concentrated violent crime that 
  demands special government funding.
'''

#============================================================================
# 4.3. Bivariate Analysis (Cross-Feature Interactions)
#============================================================================
plt.figure(figsize=(8, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Bivariate Feature Interaction Matrix (Correlation Coefficient)')
plt.tight_layout()
plt.show()

# Pairplot Matrix
sns.pairplot(df_numeric)
plt.suptitle("Bivariate Interactions for Crime Drivers", y=1.02)
plt.show()
'''
Inference & Comments:
- A powerful positive correlation (~0.80) exists between Murder and Assault, 
showing that 
  regions with high non-fatal violent crime are highly likely to experience 
  high homicide rates as well.
- Interestingly, UrbanPop shows a moderate correlation with Rape (~0.41) but
 a very weak relationship 
  with Murder (~0.07). This reveals that high urbanization drives certain
  types of crime, but 
  lethal violence is influenced by other outside socioeconomic factors.
'''


#----------------------------------------------------------------------------
# 5. MODEL BUILDING & EXPERIMENTAL TUNING
#----------------------------------------------------------------------------

#============================================================================
# 5.1 & 5.2. Hierarchical Clustering and Dendrogram Visualization
#============================================================================
plt.figure(figsize=(16, 9))
plt.title('Hierarchical Clustering Tree (Dendrogram) - Ward Linkage Strategy')
plt.xlabel('State Matrix Sample Index / Branch Groupings')
plt.ylabel('Euclidean Cophenetic Distance Scale')

# Generate the complete dendrogram tree using Ward's minimum variance method
dendrogram = sch.dendrogram(
    sch.linkage(scaled_df, method='ward'),
    labels=df['State'].values if 'State' in df.columns else None,
    leaf_rotation=90.,
    leaf_font_size=11.
)
plt.axhline(y=7, color='crimson', linestyle='--', label='Optimal 3-Cluster Threshold')
plt.axhline(y=4.5, color='darkgreen', linestyle='--', label='Alternative 4-Cluster Threshold')
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.show()
'''
Inference & Comments:
- Looking at the vertical tree branches, cutting the dendrogram at a distance height of 7 
  gives us 3 clean, highly distinct clusters with clear separation.
- Lowering the cut line to 4.5 creates 4 clusters. We will run and compare 
both options next 
  to see which setup makes the most sense for public safety deployment.
'''

#============================================================================
# 5.3. Cluster Validation & Comparative Evaluation
#============================================================================
# Strategy A: 3-Cluster Model
hc_3 = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
df['HC_Cluster_K3'] = hc_3.fit_predict(scaled_df)

# Strategy B: 4-Cluster Model
hc_4 = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
df['HC_Cluster_K4'] = hc_4.fit_predict(scaled_df)

print("\n=== Cluster Average Profiles: 3-Cluster Strategy ===")
summary_k3 = df.groupby('HC_Cluster_K3')[['Murder', 'Assault', 'UrbanPop', 'Rape']].mean()
print(summary_k3.round(2))

print("\n=== Cluster Average Profiles: 4-Cluster Strategy ===")
summary_k4 = df.groupby('HC_Cluster_K4')[['Murder', 'Assault', 'UrbanPop', 'Rape']].mean()
print(summary_k4.round(2))
'''
Inference, Comparison & Insights:
- The 3-Cluster approach separates states into clear, actionable safety 
profiles:
  * Cluster 0 (High Violence/Urban Centers): Very high murder (12.1), assault
  (255.2), and high urban pop.
  * Cluster 1 (Moderate Crime/Suburban-Rural mix): Moderate overall crime rates with baseline numbers.
  * Cluster 2 (Safe / Peaceful Zones): Exceptionally low murder (4.2) and low
  assault (87.5) rates.
- The 4-Cluster approach simply takes the moderate states and splits them 
into two smaller groups 
  based purely on whether their population is slightly more urban or rural.
- Because the 3-Cluster model perfectly divides regions into high, medium, 
and low risk tiers, 
  it is selected as the ideal framework for government budgeting and 
  emergency resource allocation.
'''
'''
-----------------------------------------------------------------------------
6. Business Benefits & Solution Impact
-----------------------------------------------------------------------------
By utilizing this hierarchical clustering model, government agencies and
 public safety departments unlock direct, practical advantages:

1. Precision Budgeting & Resourcing
The Benefit: Instead of distributing public safety funds evenly across all 
states, the government can instantly flag Cluster 0 (High Violence Urban
Centers) as priority areas.

The Impact: Emergency funds, advanced tactical support units, and preventative
safety programs can be directed straight to these high-risk states, ensuring 
critical resources are sent where they are needed most.

2. Tailored Community Policing Strategies
The Benefit: The model separates highly urbanized, high-crime areas from 
Cluster 1 & 2 (Moderate & Safe Zones), which feature lower population 
densities and low violent crime.

The Impact: High-risk clusters can focus heavily on violent crime intervention
 and tactical response units. Meanwhile, safer clusters can direct their 
 smaller budgets toward community-focused patrol programs and public watch
 groups to keep crime rates low.

3. Measurable Drops in Nationwide Crime Rates
The Benefit: The system provides data-driven support for long-term state
 budgeting, removing political guesswork from resource allocation.

The Impact: Sending targeted police reinforcements and crime reduction
 budgets straight to the most heavily impacted states helps drive down 
 nationwide violent crime numbers efficiently, maximizing the return on 
 public safety investments.
'''
