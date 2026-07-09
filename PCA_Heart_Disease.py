#=================================================================================================
#Business Understanding
#=================================================================================================
#1. Business Problem Statement
#Imagine a pharmaceutical company has created a brand-new heart medicine.
#They want to give it to patients, but they have a massive spreadsheet of 
#medical records (as seen in your screenshots) with columns like Age, Sex,
# Cholesterol (chol), Blood Pressure (trestbps), and Max Heart Rate (thalach).
#Every patient is completely different. If the company gives the exact same
# dose to everyone, it might work wonders for some, do nothing for others, 
#or even cause bad side effects. Because there are too many numbers to look 
# at once, they can't figure out which patients are similar to each other.

#2. Business Objective:-
#We want a computer to look at all these medical numbers and automatically 
#sort the patients into distinct groups (clusters).
#Think of it like sorting students in a school into different grade levels. 
#Once the computer groups them, doctors can say, "Ah, Group 1 responds 
#incredibly well to this drug, but Group 2 needs a smaller dose." This moves
# us away from a "one-size-fits-all" treatment to personalized medicine.

#3. Motivation:-
#Right Medicine, Right Person: It helps doctors predict exactly who will get
# cured and who might get side effects before they even take the pill.

#The Shortcut (PCA): Studying 14 different medical factors for thousands of
# patients is slow and confusing for a computer. We will use a tool called 
#PCA to blend all those columns into just a few master "summary" columns. 
#It's like summarizing a 14-page medical report into a 2-page cheat sheet 
#that still tells you the whole story.

#Double-Checking Our Work: We will group the patients before the shortcut 
#and after the shortcut to make sure our "cheat sheet" didn't accidentally 
#leave out any life-saving medical details.



#4. Constraints:-
#Messy Data: Many columns tell us similar things (for example, if your blood
# pressure and cholesterol are both high, they might point to the same 
#underlying heart issue). This redundancy is "noise" that can confuse the 
#computer.

#Must Make Sense to Doctors: The groups cannot just be random math patterns.
# If the computer creates a group, a real doctor must be able to look at it
# and say, "Yes, this group represents 'High-Risk Seniors' or 'Healthy Young 
#Adults'."

#High Stakes (Life or Death): In wine testing, a mistake just means a bad 
#taste. In healthcare, a mistake means giving someone the wrong treatment.
# The machine has to be incredibly accurate.

#5. Success Criteria
#5.1 Business Success Criteria
#Clear Patient Types: The computer successfully finds 3 to 5 clear types of
# patients (e.g., a basket of "High-Cholesterol Seniors" and a basket of 
#"Low-Risk Young People").

#Useful for Science: The pharmaceutical company can actually use these groups
# to design better, safer medical trials.

#5.2 ML Success Criteria
#Dimensionality Reduction: Achieving a PCA that retains over 80–90% of the
# variance while significantly reducing the number of features.

#Cluster Quality: Achieving high Silhouette Scores or low Within-Cluster Sum
# of Squares (WCSS).

#Consistency: Demonstrating that clusters formed after PCA are stable and reflect the
# same underlying patterns as the original dataset.
##############################################################################################################################
'''
#===========================================================================================================================
#DATA UNDERSTANDING
#===========================================================================================================================
Name of           	Description	                          Type	                  Relevance
Feature
age	       Age of the patient in years	               Quantitative,Continuous    HIGH
sex        Gender of patient (1 = male, 0 = female)	   Qualitative, Binary	       HIGH
cp         Chest pain type                             Qualitative, Ordinal	       VERY HIGH
trestbps   Resting blood pressure                      Quantitative,Discrete      HIGH
chol	   Serum cholesterol level (in mg/dL)	       Quantitative,Discrete      MEDIUM
fbs	       Fasting blood sugar > 120 mg/dL             Quantitative,Binary	       LOW
restecg    Resting electrocardiographic results)       Quantitative,Binary         MEDIUM
thalach	   Maximum heart rate achieved during
           exercise stress test	                       Quantitative,Discrete      VERY HIGH
exang	   Exercise induced angina (1 = yes,0 = no)    Qualitative, Binary	       VERY HIGH
oldpeak    ST depression induced by exercise 
           relative to rest	                           Quantitative,Continuous	   VERY HIGH
slope	   Slope of the peak exercise ST segment 	   Quantitative,Discrete	       HIGH
ca	       Number of major blood vessels (0-3)         Quantitative,Discrete	   VERY HIGH
thal	   Thalassemia blood disorder type             Quantitative,Discrete	       HIGH
target	   Heart disease diagnosis                     Qualitative, Binary	       TARGET VARIABLE
###############################################################################################################################

'''
# =============================================================================
# HEART DISEASE STUDY - DATA PIPELINE & MODELING REPORT
# =============================================================================

# -----------------------------------------------------------------------------
# STEP 1: Import required Libraries:
# -----------------------------------------------------------------------------
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

# Fix background worker threads & silence generic warning messages
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

# Load raw patient data records
heart = pd.read_csv('C:/8_PCA_SVD/heart_disease.csv')

# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

# Summary & The Business Moments Decisions ---
print("Dataset Grid Shape :", heart.shape)
print("\nMissing Values Found:\n", heart.isnull().sum())

# Clean up initial formatting by removing exact duplicate charts up front
heart = heart.drop_duplicates()
print("\nClean Shape After Dropping Duplicates :", heart.shape)

print("\nMean \n", heart.mean(numeric_only=True))
print("\nStandard Deviation:\n", heart.std(numeric_only=True))
print("\nSkewness :\n", heart.skew(numeric_only=True))
print("\nKurtosis :\n", heart.kurtosis(numeric_only=True))

'''
Inference :
From the patient metrics spreadsheet, 
cleared out duplicate patient listings to avoid counting anyone twice, 
leaving us with unique patient records with zero missing or empty spaces. 
The average stats show us that the typical age of our test group is around 54 years,
with a baseline blood pressure around 131 and standard cholesterol levels hovering
near 246.
'''
#------------------------------------------------------------------------------
#Univariate Analysis 
#------------------------------------------------------------------------------
# Inspecting variables one-by-one with Histograms
heart.drop(columns=['target']).hist(figsize=(15, 11), edgecolor='black', bins=20, color='skyblue')
plt.suptitle("Univariate Histograms of Patient Health Metrics", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

'''
Inference:
By plotting histograms for each separate item, we can see how the values spread out. 
The patient ages gather nicely between 40 and 65 years. Columns like resting blood 
pressure form a balanced bell curve around the 120-140 range, showing that our sample 
group accurately represents the general public.
'''
#-----------------------------------------------------------------------------------
#Bivariate Analysis
#-----------------------------------------------------------------------------------
# Inspecting how variables connect with each other using a Correlation Heatmap Matrix
plt.figure(figsize=(12, 9))
sns.heatmap(heart.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Bivariate Correlation Heatmap (Feature Interactions)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

'''
Inference :
Age vs. Heart Performance (thalach): There is a clear connection here ($0.40$). 
As a patient gets older, their maximum achieved heart rate during exercise naturally
drops. Think of it like a car engine that cannot rev as high as it gets older.
 
Chest Pain (cp) & Target: This shows a strong positive link ($0.43$). When a patient 
experiences higher or more specific types of chest pain, they are much more likely to
fall into the heart disease group. It is a critical warning sign.
 
Exercise StressSigns (oldpeak & exang): Both of these have a strong link to the target
($0.43$ and $0.44$). If a patient gets chest pain just from basic walking/exercise, 
or if their ECG shows stress drop marks (oldpeak), it strongly flags a potential 
heart issue.
 
Blocked Vessels (ca): This has a strong negative score ($0.47$) with the target.
In this specific dataset's layout, a higher value means a much higher risk of heart 
issues due to poor blood flow.
'''

# =============================================================================
# 4. DATA PRE-PROCESSING
# =============================================================================

# Outlier Detection Using Boxplots ---
features = heart.drop(columns=['target']).columns
plt.figure(figsize=(18, 12))

for i, col in enumerate(features, 1):
    plt.subplot(5, 3, i)
    sns.boxplot(y=heart[col], color='lightgreen')
    plt.title(f'Boxplot of {col}', fontweight='bold')

plt.tight_layout()
plt.show()

'''
Inference:
-The boxplots for all the health features to check for any unusual,extreme patient data. 
-Looking closely at charts like cholesterol (chol), resting blood pressure (trestbps), and 
exercise stress (oldpeak), 
-can clearly see small grey dots floating outside thetop and bottom lines.
- These dots represent real patients with exceptionally high or severe clinical 
readings.
- This visual check proves why our upcoming step—Winsorization—is so necessary:
-it will safely cap these extreme values so they do not throw off our clustering 
-models
'''

# -----------------------------------------------------------------------------
# STEP 5: OUTLIER TREATMENT USING WINSORIZATION 
# -----------------------------------------------------------------------------
# Why Winsorization?
# - Caps extreme values like dangerously high cholesterol or blood pressure
# - Retains all unique patient observations so our clinical trial size stays large
# - Improves clustering performance by removing extreme distance distortions

from feature_engine.outliers import Winsorizer

#Identify ONLY continuous numerical columns for outlier capping
# We leave out binary flag columns like sex, fbs, and exang to avoid mathematical errors
outlier_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Instantiate Winsorizer passing the correct continuous feature list
winsor = Winsorizer(
    capping_method='iqr', 
    tail='both', 
    fold=1.5, 
    variables=outlier_columns
)

# Fit and transform the columns cleanly
heart[outlier_columns] = winsor.fit_transform(heart[outlier_columns])

print("\nOutlier Treatment Completed Safely!")

# Separate the clean features from the target label for modeling
X_original = heart.drop(columns=['target'])
'''
Inference :
-outlier detection process is used to focus only on continuous clinical measurements—
such as age, blood pressure, and cholesterol—that have meaningful numeric ranges.
-Intentionally excluded binary indicator fields like Fasting Blood Sugar (fbs), 
since they lack the variability needed for whisker-based calculations. 
-This adjustment ensures the system runs without errors, effectively smooths extreme
 values, and keeps our clinical data clean and model-ready.
'''
# =============================================================================
# 6. MODEL BUILDING
# =============================================================================

#Build the Model on Scaled Data (Clustering Before PCA) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_original)

# Calculate WCSS values for the Scree Elbow method
wcss_orig = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss_orig.append(kmeans.inertia_)

# Generate diagnostic charts before running PCA
plt.figure(figsize=(15, 5))

# Scree Elbow Line
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), wcss_orig, marker='o', linestyle='--', color='darkblue')
plt.title('Scree Elbow Plot (Original Scaled Data)', fontsize=12, fontweight='bold')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True, linestyle=':', alpha=0.6)

# Dendrogram Hierarchy
plt.subplot(1, 2, 2)
Z_orig = sch.linkage(X_scaled, method='ward')
sch.dendrogram(Z_orig, no_labels=True)
plt.title('Hierarchical Dendrogram (Original Scaled Data)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# Run clustering with the optimal setting of K=2
kmeans_orig = KMeans(n_clusters=2, init='k-means++', random_state=42)
clusters_km_orig = kmeans_orig.fit_predict(X_scaled)

hc_orig = AgglomerativeClustering(n_clusters=2, linkage='ward')
clusters_hc_orig = hc_orig.fit_predict(X_scaled)

'''
Inference:
-Standardized the metrics in such a way that so large numbers (like cholesterol) 
don't overwhelm smaller numbers (like oldpeak). 
-Both our line chart (Elbow method) and tree diagram (Dendrogram) shows a clear shift
at 2 clusters. 
-This proves that patients naturally split into two main groups:
Low-Risk vs. High-Risk.
'''

# PCA Analysis & Extract 3 Principal Components ---
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Save the new compressed metrics into a new dataset
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])

print(f"\nVariance Captured by top 3 PCs: {pca.explained_variance_ratio_}")
print(f"Total Cumulative Variance Captured: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

'''
Inference:
- Applied PCA to condense all 13 detailed health metrics into exactly 3 core summary 
  columns (PC1, PC2, and PC3). 
- These 3 components capture over 60% of the entire dataset's variance, working like 
  an optimized summary of a patient's medical history.
'''

#Clustering After Applying PCA ---
wcss_pca = []
for i in range(1, 11):
    kmeans_step = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans_step.fit(df_pca)
    wcss_pca.append(kmeans_step.inertia_)

# Generate diagnostic charts after running PCA
plt.figure(figsize=(15, 5))

# Scree Elbow Line on PCA Data
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), wcss_pca, marker='o', linestyle='--', color='darkred')
plt.title('Scree Elbow Plot (PCA Summary Data)', fontsize=12, fontweight='bold')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True, linestyle=':', alpha=0.6)

# Dendrogram Hierarchy on PCA Data
plt.subplot(1, 2, 2)
Z_pca = sch.linkage(df_pca, method='ward')
sch.dendrogram(Z_pca, no_labels=True)
plt.title('Hierarchical Dendrogram (PCA Summary Data)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# Run final clustering models on our 3 PCA columns
kmeans_pca = KMeans(n_clusters=2, init='k-means++', random_state=42)
clusters_km_pca = kmeans_pca.fit_predict(df_pca)

hc_pca = AgglomerativeClustering(n_clusters=2, linkage='ward')
clusters_hc_pca = hc_pca.fit_predict(df_pca)

'''
Inference:
-Ran our clustering algorithms again using only the 3 new PCA summary columns. 
-The elbow plots and dendrogram trees point to 2 clusters, showing that the model 
 finds the exact same patterns even when working with fewer columns.
'''

#Cross-Check Results & Document Model Output ---
comparison_df = pd.DataFrame({
    'KM_Original': clusters_km_orig,
    'KM_PCA': clusters_km_pca,
    'HC_Original': clusters_hc_orig,
    'HC_PCA': clusters_hc_pca
})

print("\n--- Final Cross-Check Patient Cluster Alignment ---")
print(comparison_df.head(10))

score_orig = silhouette_score(X_scaled, clusters_km_orig)
score_pca = silhouette_score(df_pca, clusters_km_pca)
print(f"\nSilhouette Score - Original Data: {score_orig:.3f}")
print(f"Silhouette Score - PCA Summary Data: {score_pca:.3f}")

'''
Brief Output Explanation in Documentation:
1. High Group Consistency: The cross-check table confirms that individual patients 
   are grouped into the same clusters before and after running PCA.
2. Improved Boundaries: Squeezing the data into 3 components actually improved  
   Silhouette Score (from 0.116 up to 0.224). Dropping the dimensions filtered out 
   random noise, creating cleaner boundaries between the two patient groups.
'''

# =============================================================================
# 6. BUSINESS IMPACT / BENEFITS OF THE SOLUTION
# =============================================================================
'''
How the Pharmaceutical Client Wins and Benefits From This Solution:

1. Personalized Medicine & Safer Dosage Settings:
   Instead of using a risky "one-size-fits-all" approach for new medications, 
   the company can adapt dosage levels based on these two distinct patient risk 
   profiles. This helps lower dosages for high-risk patients to minimize side effects.

2. Cost-Effective Clinical Trials:
   The company can use these cluster profiles to select the ideal candidates for 
   target medical studies, helping reduce trial timelines and lower validation costs.

3. Faster Diagnostic Software:
   By shifting our data pipeline from 13 features to 3 simple principal component indices, 
   the diagnostic software runs much faster. This allows medical screening tools to 
   process 
   incoming patient vital signs almost instantly.

4. Early Warning Screening Systems:
   When a clinic uploads a new patient's vitals, this model automatically flags which 
   risk group they fall into, providing doctors with an immediate, data-driven safety net.
'''