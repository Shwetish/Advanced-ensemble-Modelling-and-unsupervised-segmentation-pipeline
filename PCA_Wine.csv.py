#=================================================================================================
#Business Understanding
#=================================================================================================
#Business problem statement:In a a massive warehouse filled with thousands of
# wine bottles, but we have no idea which ones taste or look similar.
#we have 13 different chemical columns (like Alcohol, Magnesium, Color, and 
#Proline). Trying to sort thousands of wines by looking at 13 different numbers
#at the same time is completely overwhelming for a human. Because we can't 
#group them, we can't organize our inventory, we can't recommend the right 
#wine to a customer, and we can't keep our quality consistent.

#2. Business Objective:-
#The main goal is to automatically group or "segment" these wines into 
#3 natural categories 
#(clusters) using their chemical properties. This helps in identifying which
# wines belong to which vineyard or quality tier without needing a
#human expert to taste every single bottle.

#3. Motivation
#Cost Savings: Running a few chemical tests is cheaper than hiring
#professional wine tasters for every batch.
#Consistency: Machines don't get tired; they will always group wines the 
#same way based on the data.
#Efficiency: Instead of forcing the computer to study all 13 confusing 
#columns for every single wine, we will use a tool called PCA to blend those
# 13 columns into just 3 master "summary" columns. It’s like turning a 
#13-ingredient recipe into a 3-ingredient shortcut that still tastes exactly
# the same. This saves immense computer processing time and lab resources.

#4. Constraints
#Data Scale: Some numbers (like Proline) are very large, while others 
#(like Nonflavanoids) are very small. We must "normalize" them so the big 
#numbers don't drown out the small ones.

#Hidden Patterns: We cannot use the "Type" column to help the machine; 
#it must find these groups on its own using only the chemical measurements.

#5. Success Criteria
#I. Business Success Criteria
#The Customer Test: If a customer buys two bottles from the "Group 1" basket,
# those two wines should look and taste almost identical to them.
#Real-World Reality: The 3 groups discovered by the machine must actually
# make sense and match the 3 real types of wine known in the wine industry.

#II. ML (Machine Learning) Success Criteria
#The "Elbow" Match: The Scree Plot (Elbow Method) should clearly point to 3
# clusters both before and after we simplify the data with PCA.
#High Variance: The first 3 Principal Components (PCs) should explain a 
#large chunk
#(usually over 65%) of the total information in the original 13 columns.
'''
#================================================================================================
#DATA UNDERSTANDING
#================================================================================================
Feature                      Description                            Type               Relevance
Type          The category or cultivar of the wine.Categorical    (Integer)             Low
Alcohol       Percent of alcohol content by volume.               Continuous            High
Malic         Concentration of malic acid in the wine.            Continuous            High
Ash           Total inorganic residue after evaporation.          Continuous            Medium
Alcalinity    The alkalinity of the ash content.                  Continuous            High
Magnesium     Quantity of magnesium mineral.                      Discrete              Medium
Phenols       Total quantity of phenolic compounds.               Continuous            High
Flavanoids    Specific group of plant-based antioxidants.         Continuous            High
Nonflavanoids Phenolic compounds not classified as flavanoids.    Continuous            Medium
Proanthocyanins    Specific tannins that affect dryness.          Continuous            High
Color         Intensity of the wine's color.                      Continuous            High
Hue           The shade/tint of the wine.                         Continuous            High
Dilution      Measurement of protein concentration (OD280/OD315). Continuous            High
Proline       Concentration of the amino acid Proline.            Discrete              High
################################################################################################
'''
#===========================================================================
#STEP 1: Import Required Libraries
#=============================================================================
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
#=============================================================================
#STEP 2: Load Dataset
#=============================================================================
wine = pd.read_csv("C:/8_PCA_SVD/wine.csv")

print(wine.head())
#=============================================================================
STEP3. EXPLORATORY DATA ANALYSIS (EDA)
=============================================================================
# Summary Statistics
print("Shape of Dataset :", wine.shape)
print("\nDataset Information")
print(wine.info())

print("\nMissing Values")
print(wine.isnull().sum())

print("\nDuplicate Rows :", wine.duplicated().sum())

wine = wine.drop_duplicates()

print("\nDescriptive Statistics")
print(wine.describe().T)

print("\nMean")
print(wine.mean(numeric_only=True))

print("\nStandard Deviation")
print(wine.std(numeric_only=True))

print("\nSkewness")
print(wine.skew(numeric_only=True))

print("\nKurtosis")
print(wine.kurtosis(numeric_only=True))
#Inference
'''
Dataset contains 178 wine records and 14 variables.
No missing values are present.
No duplicate observations are present.
Features have different ranges, so scaling is required.
'''
=============================================================================
# Univariate Analysis
=============================================================================
#Histograms
wine.drop(columns=['Type']).hist(
    figsize=(16,12),
    bins=20,
    edgecolor='black'
)

plt.suptitle("Distribution of Wine Variables")
plt.tight_layout()
plt.show()
'''
Inference
Most variables are approximately normally distributed.
Some variables like Magnesium and Proline are right-skewed.
Chemical properties vary widely among wines.
'''
#Boxplots
plt.figure(figsize=(18,12))

for i, col in enumerate(wine.columns[1:],1):
    plt.subplot(5,3,i)
    sns.boxplot(y=wine[col], color='lightgreen')
    plt.title(col)

plt.tight_layout()
plt.show()
'''
Inference
Some variables contain outliers.
Extreme values represent highly concentrated wines.
'''
=============================================================================
Bivariate Analysis
=============================================================================
#Correlation Heatmap
plt.figure(figsize=(12,10))

sns.heatmap(
    wine.corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Correlation Heatmap")
plt.show()
'''
Inference
Phenols and Flavanoids are highly positively correlated.
Alcohol and Proline also show strong positive correlation.
Several variables are correlated, making PCA useful.
'''
#=============================================================================
#Step 4. DATA PREPROCESSING
#=============================================================================
#Separate Features
X = wine.drop('Type', axis=1)

# Outlier Treatment
winsor = Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=X.columns.tolist()
)

X = winsor.fit_transform(X)

print("Outlier Treatment Completed")

# Feature Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

print(X_scaled.head())
'''
Inference
All variables now have equal importance.
Large features like Proline no longer dominate clustering.
'''
=============================================================================
STEP:5. MODEL BUILDING
=============================================================================
=============================================================================
 K-Means Clustering on Original Data
=============================================================================
#Elbow Method
wcss_pca = []

for i in range(1, 11):

    km = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42
    )

    km.fit(df_pca)

    wcss_pca.append(km.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss_pca, marker='o')

plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method on PCA Data")

plt.show()
'''
Inference
The elbow again occurs at K = 3.
Therefore, PCA preserved the natural grouping of wines.
'''
=============================================================================
Build K-Means Model on PCA Data
=============================================================================
kmeans_pca = KMeans(
    n_clusters=3,
    init='k-means++',
    random_state=42
)

clusters_km_pca = kmeans_pca.fit_predict(df_pca)

print(clusters_km_pca[:10])
=============================================================================
Hierarchical Clustering on PCA Data
=============================================================================
#Dendrogram
plt.figure(figsize=(15, 7))

Z_pca = sch.linkage(
    df_pca,
    method='ward'
)

sch.dendrogram(Z_pca)

plt.title("Dendrogram - PCA Data")
plt.xlabel("Observations")
plt.ylabel("Distance")

plt.show()
'''
Inference
The dendrogram also suggests approximately 3 clusters.
'''
=============================================================================
Build Hierarchical Model on PCA Data
=============================================================================
hc_pca = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

clusters_hc_pca = hc_pca.fit_predict(df_pca)

print(clusters_hc_pca[:10])
=============================================================================
Silhouette Scores
=============================================================================
#Original Data
score_orig = silhouette_score(
    X_scaled,
    clusters_km_orig
)
#PCA Data
score_pca = silhouette_score(
    df_pca,
    clusters_km_pca
)
#Print Scores
print("Silhouette Score (Original):",
      round(score_orig, 3))

print("Silhouette Score (PCA):",
      round(score_pca, 3))
'''
Inference
A positive silhouette score indicates good cluster separation.
If the PCA score is higher, PCA has created cleaner and more compact clusters.
If both scores are similar, PCA reduced dimensions without losing important information.
'''
=============================================================================
Cross-Check Cluster Results
=============================================================================
comparison = pd.DataFrame({
    'KM_Original': clusters_km_orig,
    'KM_PCA': clusters_km_pca,
    'HC_Original': clusters_hc_orig,
    'HC_PCA': clusters_hc_pca
})

print(comparison.head(10))
#Crosstab Comparison
print(
    pd.crosstab(
        clusters_km_orig,
        clusters_km_pca,
        margins=True
    )
)

print(
    pd.crosstab(
        clusters_hc_orig,
        clusters_hc_pca,
        margins=True
    )
)
'''
Inference
Cluster patterns before and after PCA are very similar.
PCA retained most of the important information in the dataset.
The wine data naturally forms 3 clusters both before and after dimensionality reduction.
'''
=============================================================================
Final Model Conclusion
=============================================================================
Before PCA
Elbow Method suggested 3 clusters.
Dendrogram also suggested 3 clusters.
Wines were successfully divided into three natural groups.
After PCA
PCA reduced 13 variables into only 3 principal components.
The first three components retained approximately 66% of the original information.
Elbow Method and Dendrogram again suggested 3 clusters.
Final Conclusion
Wine data naturally forms three clusters.
PCA reduced dimensionality and computational complexity.
Similar clustering results were obtained before and after PCA.
=============================================================================
FINAL MODEL OUTPUT DOCUMENTATION
=============================================================================
Before PCA
Elbow Method suggested 3 clusters.
Dendrogram also suggested 3 clusters.
K-Means and Hierarchical clustering successfully grouped wines into three categories.
After PCA
PCA reduced 13 variables into 3 principal components.
The first three PCs retained approximately 66% of the information.
Elbow plot and dendrogram again suggested 3 clusters.
Cluster structure remained almost unchanged.
Final Conclusion
Wine data naturally forms 3 clusters.
PCA significantly reduced dimensionality and computation time.
Similar clustering results were obtained before and after PCA.
=============================================================================
6. BUSINESS IMPACT / BENEFITS
=============================================================================
1. Better Inventory Management

Similar wines can be stored together.

2. Faster Product Classification

New wines can be automatically assigned to a category.

3. Improved Customer Recommendations

Customers can be recommended wines from the same cluster.

4. Better Quality Control

Similar chemical profiles can be monitored consistently.

5. Reduced Processing Time

PCA reduced 13 variables to only 3 components while preserving most information.

Final Business Inference

The wine dataset naturally forms three clusters. PCA successfully compressed the dataset from 13 chemical measurements to only 3 principal components while maintaining almost the same clustering structure. This enables faster, simpler, and more efficient wine segmentation for business decision-making.