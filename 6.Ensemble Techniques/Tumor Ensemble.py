#===================================================================
#Business Understanding:-
#===================================================================
#1. Business Problem StatementDoctors have to look at complex tissue
# measurements (like cell size, shape, and roughness) to determine
# if a tumor is harmless (Benign) or cancerous (Malignant). 
#Doing this manually takes a lot of time, and if a doctor 
#misinterprets a cell measurement, a patient might not receive 
#life-saving treatment quickly enough.

#2. Business ObjectiveTo build an automated screening tool that 
#instantly analyzes cell measurements from a tissue sample to 
#accurately flag whether a tumor is cancerous or benign, helping 
#doctors make faster, safer treatment decisions.

#3. MotivationEarly Detection Saves Lives: Catching a malignant 
#(cancerous) tumor early significantly increases a patient's chances
# of survival.Reduces Human Error: It provides a reliable second
# opinion for doctors so no dangerous tumor is missed due to 
#fatigue or oversight.Speeds Up Results: Instead of waiting days
# for complex manual reviews, the tool can help classify risk levels
# almost instantly, reducing patient anxiety.

#4. Constraints:-High Cost of a False Negative (Extreme Risk): If 
#the model tells a patient their tumor is harmless (Benign) when 
#it is actually cancerous (Malignant), they won't get treatment, 
#which is life-threatening. The model must avoid this mistake at 
#all costs.Complex Features: The dataset contains many highly 
#specific, correlated medical metrics 
#(like radius_mean, perimeter_mean, and area_worst), which requires
# careful feature selection so the model doesn't get confused.
#Patient Trust: Medical tools require extremely high clarity;
# doctors need to trust the predictions before using them to make
# clinical choices.5. Business Success CriteriaPrimary Goal:
#Successfully screen and correctly route at least 95% of true 
#cancer cases to immediate specialist care.Secondary Goal: Speed
# up the diagnostic pipeline in clinics, allowing doctors to 
#process patient biopsies significantly faster.

#6. ML (Machine Learning) Success Criteria:-
#Recall Score $\ge$ 95% to 98% (The Critical Metric): The model
# must have a near-perfect recall rate for the "Malignant" class 
#to guarantee that practically zero cancer cases are missed 
#(minimizing False Negatives).Overall Accuracy $\ge$ 90%: The
# model should maintain high overall precision so doctors are not
# overwhelmed by false alarms on healthy patients.

#=============================================================================
#Data Understanding:-
#=============================================================================
'''
Name of Feature 	Description      	   Type    	    Relevance
id	             Unique identification
                 number assigned to each  Quantitative,
                 patient sample.	      Nominal	    Irrelevant.
diagnosis	   The target label indicatingQualitative,
               if the tumor cell is malignant           Highly
               (M) or benign (B).	      Nominal	    Relevant 
                                                        (Target 
                                                        Variable).
radius_mean 	Distance from the center 
                of the cell nucleus to    Quantitative,  Highly
                points on its perimeter.  Continuous	 Relevant.
texture_mean 	Standard deviation of 
                gray-scale color values   Quantitative,  Highly9
                across the cell tissue.	   Continuous	 Relevant.
perimeter_mean 	Total distance around the 
                outer boundary of the cell Quantitative, Highly
                nucleus.	               Continuous	 Relevant.
area_mean 	   Total space occupied by the Quantitative, Highly
               surface of the cell nucleus.	 Continuous	 Relevant.
smoothness_   Local variation in  perimeter Quantitative,Highly
mean	     lengths or rough the edge is).	 Continuous	 Relevant.
compactness_  Calculation combining         Quantitative Highly
mean 	        perimeter and area           Continuous  Relevant
Concavity_    Severity of any dents or
mean	      indentations in the nucleus  Quantitative, Highly
              boundary.	                    Continuous	 Relevant.
points_mean   Total number of concave 
              indentations or sharp dips   Quantitative  Highly
              on the edge.	              , Continuous	 Relevant.
symmetry_     Balance score comparing
mean	     one half of the cell structure Quantitative, Highly
              to the other.	                Continuous	  Relevant.
dimension_    Fractured dimension properties
mean	      measuring the complex geometry Quantitative, Highly
              of the cell.	                 Continuous   Relevant.
'''
# ==============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# TUMOR ENSEMBLE DATASET - MALIGNANCY DETECTION & RISK PROFILING
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
# (Update this file path to match where your dataset is saved)
df = pd.read_csv("C:/18_Ensemble_Baggging/Tumor_Ensemble .csv")

print("First 5 records of the tissue biopsy screening pipeline:")
print(df.head())

# Drop non-predictive tracking features to avoid statistics distortion
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# ==============================================================================
# UNIVARIATE ANALYSIS (FOUR MOMENTS OF BUSINESS DECISION)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. First Moment - Mean (Baseline Cell Morphology)
# ------------------------------------------------------------------------------
print("\n--- FIRST MOMENT: MEAN ---")
print(df.mean(numeric_only=True).head(10))  # Showing first 10 for display brevity
'''
Inference:
The Mean provides the standard baseline anatomical dimensions of our scanned cell nuclei sample pool.

• area_mean (~654.8) -> Represents the average spatial footprint of the cell nuclei.
• radius_mean (~14.12) -> Establishes the expected distance from the center to the edge.
• concavity_mean (~0.088) -> Baseline score for structural indentations across general samples.

Used for:
Establishing diagnostic baseline benchmarks to instantly flag tissues showing abnormally 
large or irregular cellular growth.
'''

# ------------------------------------------------------------------------------
# 2. Second Moment - Variance & Standard Deviation (Diagnostic Variations)
# ------------------------------------------------------------------------------
print("\n--- SECOND MOMENT: STANDARD DEVIATION ---")
print(df.std(numeric_only=True).head(10))
'''
Inference:
High variance and spread observed in:
- area_mean / area_worst
- perimeter_mean / perimeter_worst

Business & Clinical Meaning:
The biopsy pool contains a diverse mix of structures. The high variation in cell area and 
perimeter confirms that the dataset contains a wide blend of normal, stable cellular tissue 
and highly aggressive, oversized irregular tumor abnormalities.
'''

# ------------------------------------------------------------------------------
# 3. Third Moment - Skewness (Anatomical Irregularities)
# ------------------------------------------------------------------------------
print("\n--- THIRD MOMENT: SKEWNESS ---")
print(df.skew(numeric_only=True).head(10))
'''
Inference:
Features tracking structural abnormalities (like concavity_mean, points_mean, and area_se) 
exhibit a significant Right Skew.

Business/Clinical Insight:
Most cell samples retain healthy, tight, and uniform shapes. However, a small minority of 
biopsies display extremely high deformity scores. Tree-based ensemble models handle these 
skewed structural abnormalities easily without requiring manual data transformations.
'''

# ------------------------------------------------------------------------------
# 4. Fourth Moment - Kurtosis (Structural Extreme Anomalies)
# ------------------------------------------------------------------------------
print("\n--- FOURTH MOMENT: KURTOSIS ---")
print(df.kurtosis(numeric_only=True).head(10))
'''
Inference:
Standard Error Features (_se) (High Positive Kurtosis)
Extremely leptokurtic (sharp peak). This indicates that for almost all standard cells, 
the rate of division error is consistently low and tightly grouped, but spikes dramatically 
into extreme outlier fields only when a highly active malignancy is present.
'''

# ------------------------------------------------------------------------------
# 5. Histograms (Consolidated Distribution Overviews)
# ------------------------------------------------------------------------------
# Plotting the main 'Mean' characteristics to keep evaluation view readable
mean_cols = [col for col in df.columns if '_mean' in col]
df[mean_cols].hist(figsize=(14, 10), edgecolor='black', color='crimson')
plt.suptitle("Histograms of Cell Nuclei Structural Characteristics (Mean Cohort)", fontsize=16)
plt.tight_layout()
plt.show()
'''
Inference:
Radius, Perimeter, and Area Mean:
Show classic right-tailed patterns. As cell structures expand beyond these typical 
bell-curve clusters, they indicate structural swelling—a primary clinical sign of malignancy.

Smoothness and Symmetry:
Follow highly symmetric, normal distributions, showing that core cell tissue textures 
remain relatively balanced even when sizes shift.
'''

# ------------------------------------------------------------------------------
# 6. Boxplots (Outlier & Severity Profiling)
# ------------------------------------------------------------------------------
plt.figure(figsize=(14, 6))
# Select key structural features for clean horizontal comparison scale
sns.boxplot(data=df[['radius_mean', 'texture_mean', 'perimeter_mean']], orient='h', palette='Set3')
plt.title("Boxplot of Anchor Cell Metrics (Isolating Extreme Structural Anomalies)")
plt.show()
'''
Inference:
The prominent right-side outliers across radius and perimeter represent vital clinical anomalies. 
In oncology data, these extreme data points signify advanced stages of tumor growth. They are 
genuine, critical clinical conditions and must not be deleted from your training set.
'''

# ------------------------------------------------------------------------------
# 7. Target Variable Distribution (Malignancy Diagnosis Class Balance)
# ------------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x='diagnosis', data=df, palette='bwr')
plt.title("Target Variable Distribution (Benign [B] vs Malignant [M])")
plt.show()
'''
Inference:
The target diagnostic variable shows a balanced distribution:
'B' (Benign / Healthy Tissues) → Majority Class (~63%)
'M' (Malignant / Active Cancer) → Minority Class (~37%)

Why SMOTE must be avoided here:
A 63:37 ratio is clean and easily managed by any machine learning model. Running synthetic 
sampling generation (SMOTE) here introduces a high risk of generating invalid cell coordinate 
combinations, creating model noise and reducing real-world clinical precision.
'''

# ==============================================================================
# BIVARIATE & MULTIVARIATE ANALYSIS
# ==============================================================================

# ------------------------------------------------------------------------------
# 8. Correlation Heatmap (Interdependency of Geometric Features)
# ------------------------------------------------------------------------------
plt.figure(figsize=(12, 8))
# Copy dataset and convert diagnosis string to binary map for correlation analysis
df_corr = df.copy()
df_corr['diagnosis'] = df_corr['diagnosis'].map({'M': 1, 'B': 0})

# Plotting the primary mean characteristics to maintain clean visual structure
sns.heatmap(df_corr[mean_cols + ['diagnosis']].corr(), annot=True, cmap='rocket', fmt=".2f")
plt.title("Cell Geometry & Malignancy Correlation Matrix")
plt.show()
'''
Inference:
Radius vs Perimeter vs Area (Correlation ~ 0.99):
Indicates extreme multicollinearity. Because a larger radius naturally creates a larger 
perimeter and area, these features contain redundant information.

Points Mean vs Diagnosis (Correlation ~ 0.78):
The strongest direct link to malignancy. Highlights that the presence of concave points on the 
nucleus boundary is a highly critical indicator for identifying malignant tumors.
'''

# ------------------------------------------------------------------------------
# 9. Scatter Plot (Geometric Separation Thresholds)
# ------------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(x='perimeter_mean', y='concavity_mean', hue='diagnosis', data=df, palette='bwr', alpha=0.8)
plt.title("Malignancy Zone Identification: Perimeter vs Concavity")
plt.show()
'''
Inference:
The scatter plot highlights a clear diagnostic boundary. Benign samples ('B') bunch tightly 
in the bottom-left corner (low perimeter, smooth edges). Malignant cases ('M') break away 
rapidly toward the upper-right area. This distinctive boundary allows ensemble models to classify 
the cases effectively.
'''

# ==============================================================================
# ADVANCED PROBABILITY DISTRIBUTION ANALYSIS (PDF & CDF)
# ==============================================================================

# Plotting PDF/CDF behavior for top diagnostic features (e.g., radius_mean)
key_features = ['radius_mean', 'concavity_mean']

for col in key_features:
    plt.figure(figsize=(12, 4))

    # Probability Density Function (PDF)
    plt.subplot(1, 2, 1)
    sns.kdeplot(data=df, x=col, hue='diagnosis', fill=True, common_norm=False, palette='bwr')
    plt.title(f'PDF: Structural Density of {col}')

    # Cumulative Distribution Function (CDF)
    plt.subplot(1, 2, 2)
    for diag_val in df['diagnosis'].unique():
        sub_series = df[df['diagnosis'] == diag_val][col]
        sorted_vals = np.sort(sub_series)
        y_vals = np.arange(len(sorted_vals)) / len(sorted_vals)
        plt.plot(sorted_vals, y_vals, label=f'Diagnosis: {diag_val}')
        
    plt.title(f'CDF: Cumulative Risk Percentiles for {col}')
    plt.xlabel(col)
    plt.ylabel('Cumulative Probability')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
'''
Inference:
PDF Visual Benefit:
The clear shift between the curves shows how effectively these features separate the classes. 
In the `radius_mean` PDF, the benign curve peaks early (around 12), while the malignant curve 
shifts downfield, confirming its high diagnostic value.

CDF Visual Benefit:
Serves as an operational risk threshold guide. For example, the chart indicates that nearly 
100% of benign cells stay below a radius of 17, while more than half of the malignant cells 
safely exceed this boundary.
'''

# ==============================================================================
# FINAL EDA SUMMARY
# ==============================================================================
'''
- The data structure provides strong predictive value, with perimeter deformities and 
  concave border counts serving as reliable indicators of malignancy.
- High geometric multicollinearity exists between radius, perimeter, and area fields. 
  Tree-based ensemble models naturally tolerate this structure well without destabilizing.
- High-value outliers represent critical, advanced medical conditions and must be retained 
  to ensure robust training.

Recommended Strategy:
1. Because the features utilize differing scales, apply a Random Forest or XGBoost Classifier. 
   These models handle variations in feature scales seamlessly without requiring normalization.
2. Monitor feature importance closely to drop highly redundant size columns if looking to 
   streamline model complexity.
'''

# ==============================================================================
# DATA PREPROCESSING OF
# TUMOR ENSEMBLE DATASET - 
# ==============================================================================

# ------------------------------------------------------------------------------
# Step 1: Import Required Libraries
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import skew
from feature_engine.outliers import Winsorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set styling configurations for clear diagnostic visual printouts
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------------------
# Step 2: Load Dataset
# ------------------------------------------------------------------------------
# (Update file path string to match your local project environment)
df = pd.read_csv("C:/18_Ensemble_Baggging/Tumor_Ensemble .csv")

print("="*60)
print(f"INITIAL DATA ENGINE INFRASTRUCTURE - Shape: {df.shape}")
print("="*60)
print(df.head())

# ------------------------------------------------------------------------------
# Step 3: Structural Feature Data Type Check
# ------------------------------------------------------------------------------
print("\n--- STEP 3: DATA TYPE SUMMARY ---")
print(df.dtypes.head(10))
'''
Inference:
Features consist entirely of continuous numerical dimensions mapping cellular nucleus 
structures, accompanied by a categorical string target tracking malignancy status.
'''

# ------------------------------------------------------------------------------
# Step 4: Missing Value Analysis & Handling
# ------------------------------------------------------------------------------
print("\n--- STEP 4: MISSING VALUE ANALYSIS ---")
# Strip potential trailing or leading whitespaces from column headers to eliminate KeyErrors
df.columns = df.columns.str.strip()

# Drop unique tracking identifiers if present to avoid statistical distortion
if 'id' in df.columns:
    df = df.drop(columns=['id'])

print("Missing Value Counts discovered across features:")
print(df.isnull().sum().sum())

# Clean missing records using a median strategy to protect against extreme structural variations
all_features = [col for col in df.columns if col != 'diagnosis']
imputer = SimpleImputer(strategy='median')
df[all_features] = imputer.fit_transform(df[all_features])

print("\nMissing values remaining after operational median imputation:", df.isnull().sum().sum())

# ------------------------------------------------------------------------------
# Step 5: Duplicate Record Removal
# ------------------------------------------------------------------------------
print("\n--- STEP 5: DUPLICATE ANALYSIS ---")
initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
duplicate_diff = initial_rows - df.shape[0]
print(f"Removed {duplicate_diff} duplicate tissue rows. Cleaned dataset shape: {df.shape}")

# ------------------------------------------------------------------------------
# Step 6: Target Variable Pre-Check & Balance Evaluation
# ------------------------------------------------------------------------------
print("\n--- STEP 6: TARGET CLASS DISTRIBUTION ---")

# 1. Clean all column names one more time to remove hidden spaces
df.columns = df.columns.str.strip()

# 2. EMERGENCY CHECK: Print what columns actually exist right now
print("CURRENT COLUMNS IN MEMORY:", df.columns.tolist())

# 3. Safe check and value count execution
target_feature = 'characters_strength'

if target_feature in df.columns:
    print("Target mapping multi-class representation established:")
    print(df[target_feature].value_counts())
else:
    print(f" ERROR: '{target_feature}' is completely missing from the dataframe!")
    print("Look at the CURRENT COLUMNS list above to see what it was changed to.")
'''
Inference:
The dataset shows a stable class distribution (~63% Benign, ~37% Malignant).
Because the baseline minority representation remains well populated, SMOTE is completely 
bypassed. Class balancing weights will be integrated directly into our estimators.
'''

# ------------------------------------------------------------------------------
# Step 7: Categorical Target String Encoding
# ------------------------------------------------------------------------------
print("\n--- STEP 7: CATEGORICAL TARGET ENCODING ---")

target_feature = 'characters_strength'

# Clean spaces just in case
df.columns = df.columns.str.strip()

# Check if the column is actually there to prevent crashes
if target_feature in df.columns:
    # Only encode if the column is text/object type
    if df[target_feature].dtype == 'object':
        le = LabelEncoder()
        df[target_feature] = le.fit_transform(df[target_feature].astype(str))
        print("Target encoded successfully.")
    else:
        print(f"Skipping encoding: '{target_feature}' is already numeric ({df[target_feature].dtype})!")
    
    print("\nTarget class representation distribution:")
    print(df[target_feature].value_counts())
else:
    print(f"SKIPPING STEP 7: '{target_feature}' was already separated or dropped in a prior cell.")
    print("Current columns left in df:", df.columns.tolist())
# ------------------------------------------------------------------------------
# Step 8: Outlier Detection Visualization
# ------------------------------------------------------------------------------
plt.figure(figsize=(12, 6))
# Sampling core mean features to present a clean visual layout scale
mean_features = [col for col in df.columns if '_mean' in col][:5]
sns.boxplot(data=df[mean_features], orient='h', palette='Set3')
plt.title("Visual Structural Distribution Profile Before Outlier Treatment", fontsize=14)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# Step 9: Outlier Treatment via Advanced Winsorization
# ------------------------------------------------------------------------------
print("\n--- STEP 9: OUTLIER TREATMENT (WINSORIZATION) ---")
# Apply Winsorization to cap extreme outlier ranges on both tails using a 1.5x IQR boundary
winsorizer = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=all_features)
df[all_features] = winsorizer.fit_transform(df[all_features])

print("Winsorization complete. Advanced cell growth metrics safely clamped to distribution limits.")

# ------------------------------------------------------------------------------
# Step 10: Skewness Re-Evaluation
# ------------------------------------------------------------------------------
print("\n--- STEP 10: SKEWNESS VALUES AFTER CAPPING ---")
skew_values = df[mean_features].apply(lambda x: skew(x))
print(skew_values)
'''
Inference:
Tree-based ensemble models calculate splits using relative rank orderings rather than absolute 
geometric distance vectors. Log transformations are skipped because these architectures are 
natively scale and skew invariant.
'''

# ------------------------------------------------------------------------------
# Step 11: Feature Scaling Assessment
# ------------------------------------------------------------------------------
'''
Operational Architecture Note:
Feature scaling transformations (StandardScaler / MinMaxScaler) are explicitly bypassed.
Reason:
Ensemble architectures (Random Forests, Bagging, and Boosting) operate on feature rank splits 
rather than distance matrices. This makes the models completely scale-invariant.
'''

# ------------------------------------------------------------------------------
# Step 12: Stratified Train-Test Splitting
# ------------------------------------------------------------------------------
print("\n--- STEP 12: STRATIFIED TRAIN-TEST SPLITTING ---")

# 1. EMERGENCY RELOAD: Force read the dataset so the columns are fresh in memory
# If your file is a CSV, change this to pd.read_csv("Password_Strength.csv")
df = pd.read_excel("C:/18_Ensemble_Baggging/Ensemble_password_strength .xlsx")

# 2. Re-apply your feature names extraction so X matches what you engineered earlier
df.columns = df.columns.str.strip()
df['characters'] = df['characters'].fillna("").astype(str)
df['length'] = df['characters'].apply(len)
df['digit_count'] = df['characters'].apply(lambda x: sum(c.isdigit() for c in x))
df['special_count'] = df['characters'].apply(lambda x: sum(not c.isalnum() for c in x))
df['upper_count'] = df['characters'].apply(lambda x: sum(c.isupper() for c in x))

# Drop the raw text column so it doesn't leak into your model
df = df.drop(columns=['characters'])

# 3. Perform the split safely
target_feature = 'characters_strength'

X = df.drop(columns=[target_feature])
y = df[target_feature]

print(f"Features shape: {X.shape} | Target shape: {y.shape}")

# Enforce 'stratify=y' to preserve exact class percentages across both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Split complete! Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
#==================================================================================
# Understanding the Model Outputs (The Confusion Matrix)
#==================================================================================
# The model places tissue samples into four simple categories based on its predictions:

# True Positives (TP) – Correct Catch: The sample is Malignant (M), and the model 
# correctly flags it.

# True Negatives (TN) – Correct Safe: The sample is Benign (B), and the model correctly
# clears it.

# False Positives (FP) – False Alarm: The sample is Benign, but the model flags
# it as Malignant by mistake. This causes extra stress and leads to a routine 
# follow-up biopsy to fix the mistake.

# False Negatives (FN) – Dangerous Miss: The sample is Malignant, but the model 
# calls it Benign. This is the most dangerous error. The patient leaves the clinic 
# without critical cancer treatment, allowing the tumor to grow and spread.

# Why "Recall" Matters Most: While overall accuracy is nice, Recall is the most 
# critical metric. High recall means the model is excellent at reducing False 
# Negatives (Dangerous Misses), ensuring no malignant tumor goes unnoticed.


# ==============================================================================
# ADVANCED PIPELINE: END-TO-END DATA PROCESSING & MULTI-ENSEMBLE OPTIMIZATION
# TUMOR ENSEMBLE DATASET 
# ==============================================================================

# ------------------------------------------------------------------------------
# Step 1: Import Core Engineering Framework
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Model Architectures
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, BaggingClassifier, 
                              AdaBoostClassifier, VotingClassifier, StackingClassifier)
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Evaluation Framework
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Global Plotting Configuration
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------------------
# Step 2: Sourcing Dataset & Environmental Grounding
# ------------------------------------------------------------------------------
# Fixed: Added the missing " before C:/
df = pd.read_csv("C:/18_Ensemble_Baggging/Tumor_Ensemble .csv")
df.columns = df.columns.str.strip()

if 'id' in df.columns:
    df = df.drop(columns=['id'])
target_feature = 'diagnosis'

print("="*80)
print(f"INITIAL CLEANING PIPELINE ENGINE INITIALIZED - Initial Shape: {df.shape}")
print("="*80)

# ------------------------------------------------------------------------------
# Step 4: Missing Value Analysis & Handling
# ------------------------------------------------------------------------------
all_features = [col for col in df.columns if col != target_feature]
imputer = SimpleImputer(strategy='median')
df[all_features] = imputer.fit_transform(df[all_features])

# Remove repeat/accidental logging instances
df.drop_duplicates(inplace=True)

# Map target string outcomes to standard machine learning flags
df[target_feature] = df[target_feature].map({'M': 1, 'B': 0})

# ------------------------------------------------------------------------------
# Step 9: Outlier Treatment via Multi-Feature Winsorization
# ------------------------------------------------------------------------------
winsorizer = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=all_features)
df[all_features] = winsorizer.fit_transform(df[all_features])

# Stratified Train-Test Splitting to secure outcome class balance across groups
X = df.drop(columns=[target_feature])
y = df[target_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------------------------------------------------------
# MODEL BUILDING ON SCALED DATA (MULTIPLE OPTIONS)
# ------------------------------------------------------------------------------
# Option A: StandardScaler (Z-Score Normalization)
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

# Option B: MinMaxScaler (Bounded Normalization between 0 and 1)
scaler_mm = MinMaxScaler()
X_train_mm = scaler_mm.fit_transform(X_train)
X_test_mm = scaler_mm.transform(X_test)

print("Scaler Matrix Option A (StandardScaler) and Option B (MinMaxScaler) Computed.")
'''
Inference:
Distance-sensitive algorithms require matching feature ranges. While tree architectures 
are natively scale-invariant, compiling models using scaled inputs guarantees cross-algorithm 
stability and satisfies strict optimization pipeline standards.
'''

# ------------------------------------------------------------------------------
# MULTI-ENSEMBLE DESIGN WITH GRIDSEARCH CV
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("             EXECUTING GRIDSEARCH CV HYPERPARAMETER TUNING")
print("="*80)

# --- A. BAGGING ENSEMBLE (Bootstrapped Decision Trees) ---
print("\n[Optimizing Bagging Ensemble...]")
bag_base = BaggingClassifier(estimator=DecisionTreeClassifier(class_weight='balanced'), random_state=42)
bag_params = {
    'n_estimators': [50, 100],
    'max_samples': [0.8, 1.0],
    'max_features': [0.8, 1.0]
}
grid_bag = GridSearchCV(bag_base, bag_params, cv=3, scoring='accuracy', n_jobs=-1)
grid_bag.fit(X_train_std, y_train)
best_bag = grid_bag.best_estimator_

# --- B. ADABOOST ENSEMBLE (Adaptive Boosting) ---
print("[Optimizing AdaBoost Ensemble...]")
ada_base = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'), random_state=42)
ada_params = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1, 1.0]
}
grid_ada = GridSearchCV(ada_base, ada_params, cv=3, scoring='accuracy', n_jobs=-1)
grid_ada.fit(X_train_std, y_train)
best_ada = grid_ada.best_estimator_

# --- C. XGBOOST ENSEMBLE (Extreme Gradient Boosting) ---
print("[Optimizing XGBoost Ensemble...]")
pos_weight_ratio = (len(y_train) - sum(y_train)) / sum(y_train)
xgb_base = XGBClassifier(scale_pos_weight=pos_weight_ratio, random_state=42, eval_metric='logloss')
xgb_params = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}
grid_xgb = GridSearchCV(xgb_base, xgb_params, cv=3, scoring='accuracy', n_jobs=-1)
grid_xgb.fit(X_train_std, y_train)
best_xgb = grid_xgb.best_estimator_

# --- D. VOTING ENSEMBLE (Soft Probability Consensus) ---
print("[Compiling Voting Consensus Classifier...]")
voting_ensemble = VotingClassifier(
    estimators=[('bag', best_bag), ('ada', best_ada), ('xgb', best_xgb)],
    voting='soft'
)
voting_ensemble.fit(X_train_std, y_train)

# --- E. STACKING ENSEMBLE (Meta-Learner Architecture) ---
print("[Compiling Stacking Meta-Learner Layer...]")
stacking_ensemble = StackingClassifier(
    estimators=[('bag', best_bag), ('ada', best_ada), ('xgb', best_xgb)],
    final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
    cv=3
)
stacking_ensemble.fit(X_train_std, y_train)

# ------------------------------------------------------------------------------
#  METRIC LOGGING & CONFUSION MATRIX COMPARISON
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("             FINAL MODEL EVALUATION ENGINE & DOCUMENTATION PRINT")
print("="*80)

model_dictionary = {
    "Bagging Classifier Ensemble": best_bag,
    "AdaBoost Classifier Ensemble": best_ada,
    "XGBoost Classifier Ensemble": best_xgb,
    "Soft Voting Classifier Consensus": voting_ensemble,
    "Stacking Meta-Classifier Model": stacking_ensemble
}

for name, model in model_dictionary.items():
    test_predictions = model.predict(X_test_std)
    acc = accuracy_score(y_test, test_predictions)
    cm = confusion_matrix(y_test, test_predictions)
    
    print(f"\n▶ ARCHITECTURE: {name}")
    print(f"   Validated Accuracy Score: {acc:.4f}")
    print("   Confusion Matrix Realization Matrix:")
    print(f"    [[ True Negatives (Benign): {cm[0][0]}   False Positives (Type I): {cm[0][1]} ]")
    print(f"     [ False Negatives (Missed): {cm[1][0]}   True Positives (Malignant): {cm[1][1]} ]]")
    print("\n   Granular Classification Performance Analysis Report:")
    print(classification_report(y_test, test_predictions, zero_division=0))
    print("=" * 70)

#==================================================================================
# Business Benefits & Impact (How the Client Wins)
#==================================================================================
#Implementing this system helps a hospital or clinic in four major ways:

#1. Early Prevention Saves Money: It moves the clinic from treating expensive, 
#advanced cancer emergencies to catching malignancy early. Early medical choices 
#or operations are much cheaper than managing long-term terminal illness.

#2. Speeds Up Clinic Workflows: Pathologists don't have to manually dig through 
#complex cell measurements during busy hours. The AI instantly connects hidden clues 
#(like matching mean radius with perimeter and concavity values) to give a risk score.

#3. Smarter Resource Use: The model automatically highlights the highest-risk 
#patients. This tells hospital managers exactly where to send specialized oncology 
#doctors, surgery setups, and critical biopsy kits, cutting down on waste.

#4. Avoids Penalties & Lowers Risk: Catching tumors early means better patient 
#recovery tracking, fewer treatment mistakes, and full compliance with healthcare 
#rules. This saves the business from massive operational legal risks and fines.
