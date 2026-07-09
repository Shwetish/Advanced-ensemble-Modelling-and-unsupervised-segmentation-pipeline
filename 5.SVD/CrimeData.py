#----------------------------------------------------------------------------
#Business Understanding:-
#----------------------------------------------------------------------------

#1. Business Problem Statement:-
#Government leaders and police departments do not know which states are the most 
#dangerous, what types of crimes dominate different regions, or if crowded cities 
#actually cause higher crime rates. Because of this, they waste tax money sending 
#police and funding to the wrong places.

#2. Business Objective:-
#To make states safer by grouping similar states together based on their crime
# profiles so that governments can allocate police budgets, resources, and 
#prevention programs efficiently.

#3. Motivation:-
#Public safety budgets are limited. Instead of treating all 50 states the 
#exact same way, the government wants to find patterns. For example, if a 
#group of states has high assault rates but low murder rates, they need 
#community policing, not maximum-security prisons.

#4. Constraints:-
#Data Limits: The dataset only tracks arrests, not the total number of actual
# crimes committed (many crimes go unreported).
#No Label: The data does not have a "Safety Grade" column (like Good/Bad). 
#We have to find hidden groups on our own using Unsupervised Learning.

# 5.Success Criteria:-
#5.1 Business Success Criteria:-
#Smart Budgeting: Successfully creating 3 to 4 distinct "State Profiles" 
#(e.g., High Crime/Highly Urban, Low Crime/Rural) so policymakers can design 
#custom safety plans for each group.
#Crime Reduction: A drop in overall national arrest numbers over the next few 
#years because resources were sent exactly where they were needed most.

#5.2 ML (Machine Learning) Success Criteria:-
#For Clustering (Grouping States via K-Means/Hierarchical Clustering): 
#Achieving a high Silhouette Score (greater than 0.50), which means the 
#states inside a group are highly similar to each other, and the groups are
# clearly separated.
#For Dimensionality Reduction (Simplifying Data via PCA): The first two main
#components must capture over 80% of the data's total variance, allowing us 
#to map all 50 states on a simple, easy-to-read 2D chart.

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
# Full US State Crime Profiling Pipeline
# Advanced EDA, 4-Moment Analysis, Standardization, & K-Means Optimization
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
from sklearn.metrics import silhouette_score

# Set aesthetic visual configurations
sns.set_theme(style="whitegrid")

#============================================================================
# 2. Load Dataset & Schema Rectification
#============================================================================
# Update this path string to map directly to your local file layout
file_path = "C:/10_Clustering/crime_data.csv"

# Reading the dataset and renaming the unlabelled State column
df = pd.read_csv(file_path)
if df.columns[0] == "Unnamed: 0" or df.columns[0] == " ":
    df.rename(columns={df.columns[0]: 'State'}, inplace=True)
else:
    df.rename(columns={df.columns[0]: 'State'}, inplace=True)

#----------------------------------------------------------------------------
# 3. Exploratory Data Analysis (EDA)
#----------------------------------------------------------------------------

#============================================================================
# 3.1. Basic Exploration & Statistical Summary
#============================================================================
print("--- First 5 Records of US Crime Dataset ---")
print(df.head())
print("\n--- Last 5 Records of US Crime Dataset ---")
print(df.tail())
print("\n--- Dataset Dimensional Shape ---")
print(df.shape)
print("\n--- Structural Information & Data Types ---")
print(df.info())
print("\n--- Descriptive Summary Metrics ---")
print(df.describe().T)
'''
Inference:
- The dataset captures 50 rows corresponding exactly to the 50 US States.
- Measures vary wildly in scale: 'Assault' values climb as high as 337, while 
  'Murder' caps out at 17.4. 
- Due to these scaling imbalances, distance calculations will over-index on 
  Assault rates unless standardization is applied.
'''

#============================================================================
# 3.2. Missing Value & Duplicate Diagnostics
#============================================================================
print("\n--- Missing Value Counts Per Feature ---")
print(df.isnull().sum())

print(f"\nTotal Duplicate State Records Found: {df.duplicated().sum()}")
'''
Inference:
- The dataset is fully intact with zero missing entries. No imputation is required.
- No duplicate records exist, confirming that each row represents a unique state.
'''

#============================================================================
# 3.3. First Four Moments of Numerical Columns
#============================================================================
numeric_cols = ['Murder', 'Assault', 'UrbanPop', 'Rape']

print("\n--- First Moment: Expected Means ---")
print(df[numeric_cols].mean())

print("\n--- Second Moment: Standard Deviation ---")
print(df[numeric_cols].std())

print("\n--- Third Moment: Skewness Matrix ---")
print(df[numeric_cols].skew())

print("\n--- Fourth Moment: Kurtosis Matrix ---")
print(df[numeric_cols].kurt())
'''
Inference:
- 'Murder' and 'Assault' exhibit low skewness, meaning their metrics are distributed
 relatively evenly across states.
- All kurtosis values are negative, indicating a light-tailed, flat distribution 
(platykurtic).
- There is a lack of extreme single-point anomalies compared to typical corporate
 customer datasets.
'''

#============================================================================
# 3.4. Univariate Analysis (Visual Distributions)
#============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(df['Murder'], kde=True, ax=axes[0,0], color='darkred')
axes[0,0].set_title('Murder Arrest Rates Distribution')

sns.histplot(df['Assault'], kde=True, ax=axes[0,1], color='orangered')
axes[0,1].set_title('Assault Arrest Rates Distribution')

sns.histplot(df['UrbanPop'], kde=True, ax=axes[1,0], color='navy')
axes[1,0].set_title('Urban Population Percentage Distribution')

sns.histplot(df['Rape'], kde=True, ax=axes[1,1], color='purple')
axes[1,1].set_title('Rape Arrest Rates Distribution')

plt.suptitle("Univariate Frequency Profiles of US State Crime Indicators", fontsize=16)
plt.tight_layout()
plt.show()
'''
Inference:
- 'UrbanPop' displays a fairly balanced, near-normal curve, indicating a uniform 
spread of urbanization across the country.
- The crime metrics show distinct multi-modal bumps, suggesting hidden structural 
subgroups (clusters) within the data.
'''

#============================================================================
# 3.5. Bivariate Analysis (Correlations & Spatial Interactions)
#============================================================================
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='Reds', fmt=".2f")
plt.title('Correlation Matrix of State Crime Indicators')
plt.tight_layout()
plt.show()

# Pairplot Matrix to observe visible cluster boundaries
sns.pairplot(df[numeric_cols])
plt.suptitle("Bivariate Interactions of Crime Metrics Across States", y=1.02)
plt.show()
'''
Inference:
- 'Assault' and 'Murder' exhibit a very strong positive correlation (0.80). 
- States with high rates of physical assaults consistently experience high homicide
 tracking metrics.
- 'UrbanPop' shows a weak direct correlation with violent crime rates, indicating 
that urban density alone does not dictate violent crime.
'''

#============================================================================
# 3.6. Outlier Isolation using Interquartile Range (IQR)
#============================================================================
print("\n--- Mathematical Outlier Volume Counts via IQR ---")
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
- Only 'Rape' displays any outlier presence (less than 2 states total).
- Because the data represents a fixed geographic census rather than volatile
 transactional entries, 
  severe outlier distortions are virtually nonexistent.
'''

#----------------------------------------------------------------------------
# 4. Data Pre-Processing
#----------------------------------------------------------------------------

#============================================================================
# 4.1. Feature Isolation & Scaling (Standardization)
#============================================================================
# Isolating spatial coordinates (removing text identifiers)
df_features = df.drop(columns=['State'])

# Initiating scale normalization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

scaled_df = pd.DataFrame(scaled_data, columns=df_features.columns)

print("\n--- Standardized Feature Calculated Means (Target ~0) ---")
print(scaled_df.mean().round(4))

print("\n--- Standardized Feature Calculated Deviations (Target ~1) ---")
print(scaled_df.std().round(4))
'''
Inference:
- All features are successfully balanced with a mean of 0 and standard deviation of 
1.
- This uniform variance ensures K-Means treats urban population density and violent
 crime indexes with equal mathematical weight.
'''

#----------------------------------------------------------------------------
# 5. Model Building & Dimensionality Reduction using SVD
#----------------------------------------------------------------------------
from sklearn.decomposition import TruncatedSVD

# 5.1 Initialize and Fit SVD
# We compress the 4 original features into 2 primary components as requested by ML Success Criteria
svd = TruncatedSVD(n_components=2, random_state=42)
svd_elements = svd.fit_transform(scaled_df)

# Store the compressed continuous state coordinates into a new DataFrame
df_svd = pd.DataFrame(svd_elements, columns=['SVD_Component_1', 'SVD_Component_2'])
df_svd['State'] = df['State'].values

#============================================================================
# 5.2 SVD Mathematical Validation (Explained Variance)
#============================================================================
print("--- Explained Variance Ratio Per Component ---")
print(svd.explained_variance_ratio_)
print(f"\nTotal Information Captured by 2 Components: {svd.explained_variance_ratio_.sum().round(4) * 100}%")

# Create a Loadings Matrix to map original attributes back to our new components
loadings = pd.DataFrame(svd.components_.T, columns=['Component_1', 'Component_2'], index=df_features.columns)

print("\n=== Feature Loadings: Mapping the Hidden Themes ===")
print(loadings.round(3))

#============================================================================
# 5.3 Visualizing the SVD State Landscape
#============================================================================
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_svd, x='SVD_Component_1', y='SVD_Component_2', color='darkred', s=100)

# Annotate a few states on the chart to see spatial distribution
for i, txt in enumerate(df_svd['State']):
    if i % 5 == 0: # Annotate every 5th state to keep the visualization clean
        plt.annotate(txt, (df_svd['SVD_Component_1'].iloc[i], df_svd['SVD_Component_2'].iloc[i]), xytext=(5,2), textcoords='offset points')

plt.title('2D Structural Map of US States via SVD Space Reduction')
plt.xlabel('SVD Component 1 (Severity of Violent Crime Profile)')
plt.ylabel('SVD Component 2 (Urbanization Index Profile)')
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.show()
#============================================================================
#6.Business Impact 
#============================================================================
1. No More Wasted Tax Money (Precision Budgets)
The Old Way: Sending identical safety budgets to every state based purely on
 population sizes.

The SVD Way: The government can see the exact coordinates of a state. They can stop 
over-funding highly populated states that are actually peaceful, and direct money 
precisely where the violence index is spiking.

2. Matching the Right Strategy to the State
SVD splits the country into two completely different problem areas, requiring 
completely different solutions:

High Violence + High City Score: These are dangerous metropolitan areas. The 
government knows to spend tax dollars on advanced city surveillance, gang task
 forces, and precinct technology.

High Violence + Low City Score: These are dangerous rural areas. Sending massive 
city SWAT teams wont work here. The government knows to spend money on state trooper 
highway patrols, regional substance abuse centers, and local community policing.

3. Clear Proof for Political Decisions
The Benefit: SVD mathematically proves that a high city population does not
 automatically cause high violent crime.

The Action: Governors and police chiefs can use this visual, data-backed 2D map to 
easily explain to lawmakers and the public exactly why they are shifting money away
from traditional prison funding and into specific local safety programs.

