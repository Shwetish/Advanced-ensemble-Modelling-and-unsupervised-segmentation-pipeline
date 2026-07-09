# ------------------------------------------------------------------------------
# BUSINESS UNDERSTANDING
# ---------------------------------------------------------------------
#1. Business Problem Statement:-
#Tax fraud and under-reporting income cost organizations and governments massive 
#amounts of money. Manually reviewing every individual profile to find tax evaders
# is like looking for a needle in a haystack—it takes too much time, costs too much 
#money, and relies too heavily on guesswork.

#2. Business Objective:-
#To automatically flag whether a tax filer's profile is Risky (taxable income $\le 30,000$) or Good 
#by analyzing basic background data like their marital status, population bracket,
# and work experience.

#3. MotivationCatch Fraud Early: Identify risky individual tax
# records automatically before processing or approving them.Smart Resource Use: Let
# fraud investigators focus their limited time on high-risk profiles instead of 
#picking files at random.Stop Financial Losses: Prevent lost revenue from tax evasion
# and uncollected duties.

#4. Constraints:-Imbalanced Classes: The data is heavily skewed because most citizens
# are honest ("Good" payers) and only a small fraction are "Risky". The model must 
#not ignore the minority fraud class.Avoid False Accusations: Aggressively accusing
# an honest client of fraud damages relationships and trust.

#5. Business Success Criteria:-
#Correctly flag high-risk tax profiles so auditing teams
# can review them immediately.Reduce the amount of lost tax revenue by closing 
#evasion gaps.

#6. ML Success Criteria:-
#High Recall for 'Risky' Profiles: The algorithm must catch as many true fraud/risky
# cases as possible, ensuring evaders don't slide through undetected.Achieve stable 
#test performance using cross-validation and ensembles (Random Forest) so the system
# handles variations in individual data without over-memorizing the clean records.

#====================================================================
#Data Understanding:-
#====================================================================
'''
Name of Feature         Description             Type        Relevance
Undergrad        Shows if the person has an    Qualitative,  Low
                 undergraduate degree(YESorNO). Nominal     Relevance
Marital          The person's relationship status Qualitative,Partially
Status          (Single, Married, or Divorced).  Nominal    Relevant
Taxable.Income   The person's yearly income    Quantitative,Highly 
                  that can be taxed.          Continuous    Relevant 
                                                            / (Target
                                                            Variable)
City.Population   The total number of people  Quantitative,  Low
                 living in the person's city. Continuous    Relevance
Work.Experience   The number of years the     Quantitative  Highly
                  person has been working.    Discrete      Relevant
Urban            Shows if the person lives    Qualitative,   Low
                 in a city area (YES or NO). Nominal        Relevance
'''
# ==============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# Fraud_Check - Tax Compliance Risk Analysis
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("c:/16_Decision_Tree/Fraud_check.csv")

df.head()

#--------------------------------------------------------------------------
# 1 First Moment - Mean
#--------------------------------------------------------------------------
df.mean(numeric_only=True)
'''
Inference:
Mean represents the average financial and experiential baseline metrics across the audited cohort.

• Average Taxable.Income -> Typical earning level of the registry.
• Average City.Population -> Medium density municipal environment.
• Average Work.Experience -> General tenure baseline for taxpayers.

Used for:
Establishing macroeconomic baselines for tax audit brackets.
'''

#----------------------------------------------------------------------------
# 2 Second Moment - Variance & Standard Deviation
#----------------------------------------------------------------------------
df.var(numeric_only=True)
df.std(numeric_only=True)
'''
Inference:
High variance observed in:
- City.Population
- Taxable.Income

Business Meaning:
Population sizes span vastly from rural outposts to dense metropolis locations.
Taxable incomes show extreme variance, implying a wide mix of low, middle, and ultra-high earners.
'''

#-----------------------------------------------------------------------------
# 3 Third Moment - Skewness
#-----------------------------------------------------------------------------
df.skew(numeric_only=True)
'''
Feature                 Skewness         Business Insight
---                     ---              ---
Taxable.Income          Near symmetric   Relatively uniform spread across income levels.
City.Population         Near symmetric   Balanced demographic sampling of city sizes.
Work.Experience         Slight skew      Relatively uniform spread of career tenures.

Inference:
The distributions are broadly flat and rectangular (near-uniform).
Tree-based partition splits operate flawlessly on uniform densities without scaling dependencies.
'''

#------------------------------------------------------------------------------
# 4 Fourth Moment - Kurtosis
#------------------------------------------------------------------------------
df.kurtosis(numeric_only=True)
'''
Inference:
Taxable.Income (-1.22)
Highly platykurtic (flat peak), indicating an evenly distributed range of income classes without a massive central cluster.

City.Population (-1.21)
Platykurtic distribution, reflecting a highly controlled, flat distribution across populations.

Work.Experience (-1.23)
Flat, rectangular kurtosis distribution curve, indicating even workforce sampling from entry-level to seniors.
'''

#----------------------------------------------------------------------------------------------
# Step 8: Histograms (Distribution Analysis)
#----------------------------------------------------------------------------------------------
df.hist(figsize=(14,10), edgecolor='black')
plt.suptitle("Histogram of All Numerical Features")
plt.show()
'''
Taxable.Income:
Shows a flat, uniform shape, indicating even distribution across income bands.
City.Population:
Evenly distributed across the cohort, proving consistent municipal exposure profiles.
Work.Experience:
Uniformly balanced across different levels of professional tenure.
''' 

#----------------------------------------------------------------------------------------------   
# Step 9: Boxplots (Outlier Detection)
#----------------------------------------------------------------------------------------------
plt.figure(figsize=(14,6))
sns.boxplot(data=df.select_dtypes(include=np.number), orient='h')
plt.title("Boxplot of All Numerical Features")
plt.show()
'''
Inference:

Taxable.Income:
No outliers observed. The distribution stays contained within expected regulatory parameters.

City.Population:
No outliers present, indicating standard metropolitan sampling scales.

Work.Experience:
Zero outliers; tracking bounded career paths between 0 to 30 years cleanly.

Business Interpretation:
Since no severe mathematical outliers exist, data distributions reflect structural uniformity.
Tree models can establish clean classification decision nodes.
'''

#---------------------------------------------------------------------------------------------
# Step 10: Target Variable Distribution (Tax Compliance Category)
#--------------------------------------------------------------------------------------------
# Create Tax Compliance Category based on problem statement
df['Tax_Risk'] = np.where(df['Taxable.Income'] <= 30000, 'Risky', 'Good')

sns.countplot(x='Tax_Risk', data=df)
plt.title("Tax Risk Category Distribution")
plt.show()
'''
Inference:
The target classes are moderately imbalanced:
Good → Majority class (~79%)
Risky → Minority class (~21%)

This represents a mild to moderate corporate imbalance, not a severe one.
SMOTE is NOT required.

Why SMOTE is a bad idea here:
Tree models handle moderate class variance seamlessly using threshold splits.
SMOTE could introduce synthetic financial records that distort true boundary edges, causing overfitting.

Best practice for this dataset:
Implement class weight optimization parameters natively within the estimators.
'''

#----------------------------------------------------------------------------------------------
# Step 11: Correlation Heatmap
#----------------------------------------------------------------------------------------------
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
'''
Taxable.Income vs Work.Experience (-0.0018)
Virtually zero linear correlation; income is driven by multi-layered business factors rather than simple tenure years.
Taxable.Income vs City.Population (-0.064)
Extremely weak inverse association; geographic size has negligible influence on income brackets.
City.Population vs Work.Experience (0.013)
No tracking relationship between structural location scales and individual career tenures.
'''

#------------------------------------------------------------------------------
# Step 12: Scatter Plot (Business Relationship)
#------------------------------------------------------------------------------
sns.scatterplot(x='City.Population', y='Taxable.Income', hue='Tax_Risk', data=df)
plt.title("City Population vs Taxable Income")
plt.show()
'''
Inference:
Isolates a clean, horizontal boundary split at the 30,000 baseline across all population ranges.
'''

#------------------------------------------------------------------------------
# Step 13: PDF & CDF Analysis
#------------------------------------------------------------------------------
for col in df.select_dtypes(include=np.number).columns:
    plt.figure(figsize=(12,4))

    # PDF
    plt.subplot(1,2,1)
    sns.kdeplot(df[col], fill=True)
    plt.title(f'PDF of {col}')

    # CDF 
    plt.subplot(1,2,2)
    sorted_vals = np.sort(df[col])
    y_vals = np.arange(len(sorted_vals)) / len(sorted_vals)
    plt.plot(sorted_vals, y_vals)
    plt.title(f'CDF of {col}')
    plt.show()
'''
Inference:
PDF highlights high, flat data distributions across the feature space.
CDF acts as a metric calculator; showing that roughly ~20% of the population naturally lands below the 30,000 threshold.
'''

#------------------------------------------------------------------------------
# Step 14: Pairplot (Feature Interaction)
#------------------------------------------------------------------------------
sns.pairplot(df[['Taxable.Income', 'City.Population', 'Work.Experience', 'Tax_Risk']], hue='Tax_Risk')
plt.show()
'''
Inference:
Risky classifications are evenly layered along the lower threshold of the feature space.

# Step 15: Final EDA Summary
- Core features are uniformly distributed and highly structured.
- Zero outliers are found across continuous numeric parameters.
- Income threshold splitting maps a clean classification split.

Suitable Models:
Decision Tree Classifier
Random Forest Classifier
'''
Phase
# ==============================================================================
# DATA PREPROCESSING & MODEL BUILDING ON FRAUD CHECK DATA
# ==============================================================================

# ---------------------------------------------------------
# Step 1: Import Required Libraries
# ---------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import skew
from feature_engine.outliers import Winsorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set(style="whitegrid")

#----------------------------------------------------------------------------
# Step 2: Load Dataset
#----------------------------------------------------------------------------
df = pd.read_csv("c:/16_Decision_Tree/Fraud_check.csv")

print("Initial Shape:", df.shape)
df.head()
'''
Inference:
Dataset contains economic and demographic taxonomy profiles for compliance analysis.
'''

#--------------------------------------------------------------------------
# Step 3: Data Type Check
#--------------------------------------------------------------------------
df.dtypes
'''
Inference:
Mixed feature space (categorical indicators mixed with numeric entries).
'''

#------------------------------------------------------------------------
# Step 4: Missing Value Analysis
#-------------------------------------------------------------------------    
print("Missing values:\n", df.isnull().sum())

#---------------------------------------------------------
# Step 5: Duplicate Removal
#---------------------------------------------------------
df.drop_duplicates(inplace=True)
print("After removing duplicates:", df.shape)
'''
Inference:
Guarantees clean, unique audited records to minimize learning distortions.
'''

#---------------------------------------------------------
# Step 6: Target Variable Discretization (Tax Risk Status)
#---------------------------------------------------------
# Problem Rule: Taxable.Income <= 30000 -> Risky, else Good
df['Tax_Risk'] = np.where(df['Taxable.Income'] <= 30000, 'Risky', 'Good')

print("Target Class Counts:\n", df['Tax_Risk'].value_counts())
'''
Inference:
Classes exhibit a moderate 79:21 structural distribution.
SMOTE is skipped to prevent distortion. We utilize 'class_weight' instead.
'''

#--------------------------------------------------------------------------------
# Step 7: Encoding Categorical Variables
#-------------------------------------------------------------------------------
le = LabelEncoder()

df['Undergrad']     = le.fit_transform(df['Undergrad'])
df['Marital.Status'] = le.fit_transform(df['Marital.Status'])
df['Urban']          = le.fit_transform(df['Urban'])
df['Tax_Risk']       = le.fit_transform(df['Tax_Risk'])  # Good -> 0, Risky -> 1

'''
Why Label Encoding?
Tree models segment numerical ranking thresholds seamlessly without requiring one-hot column generation.
'''

# ---------------------------------------------------------
# Step 8: Outlier Detection (Boxplots)
# ---------------------------------------------------------
plt.figure(figsize=(12,6))
sns.boxplot(data=df.select_dtypes(include=np.number), orient='h')
plt.title("Boxplot Before Outlier Treatment")
plt.show()
'''
Observation:
Zero statistical outliers detected across features.
'''

# ---------------------------------------------------------
# Step 9: Outlier Treatment (Winsorization Block)
# ---------------------------------------------------------
# Skipped because no features fall outside the IQR boundaries.
'''
Inference:
Dataset is structurally clean; capping steps are omitted.
'''

# ---------------------------------------------------------
# Step 10: Skewness Detection
# ---------------------------------------------------------
num_cols = df.select_dtypes(include=np.number).columns
skew_values = df[num_cols].apply(lambda x: skew(x))
print("Skewness Scores:\n", skew_values)
'''
Inference:
Near-zero skew metrics indicate uniformly flat profiles. Tree structures navigate this directly.
'''

# ---------------------------------------------------------
# Step 11: Feature Scaling (Note Block)
# ---------------------------------------------------------
'''
Feature scaling is NOT required for:
Decision Tree / Random Forest
Reason:
Tree splits are scale-invariant.
'''

# ---------------------------------------------------------
# Step 12: Train-Test Split Setup
# ---------------------------------------------------------
# Drop continuous target source column and encoded target label from feature matrix
X = df.drop(columns=['Taxable.Income', 'Tax_Risk'])
y = df['Tax_Risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train Shape: {X_train.shape} | Test Shape: {X_test.shape}")

# ---------------------------------------------------------
# Step 13: Build and Evaluate Decision Tree Model
# ---------------------------------------------------------
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42, class_weight='balanced')
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

print("\n" + "="*50)
print("DECISION TREE EVALUATION METRICS")
print("="*50)
print("Accuracy Score:", accuracy_score(y_test, dt_preds))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, dt_preds))
print("\nClassification Report:\n", classification_report(y_test, dt_preds))

# ---------------------------------------------------------
# Step 14: Build and Evaluate Random Forest Model
# ---------------------------------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("\n" + "="*50)
print("RANDOM FOREST EVALUATION METRICS")
print("="*50)
print("Accuracy Score:", accuracy_score(y_test, rf_preds))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, rf_preds))
print("\nClassification Report:\n", classification_report(y_test, rf_preds))

# ==============================================================================
# FRAUD CHECK DATA PREPROCESSING, MODEL BUILDING & ENSEMBLES
# ==============================================================================

# ---------------------------------------------------------
# STEP 1: Import Libraries & Load Data
# ---------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score

# Assuming df is loaded from your CSV path (e.g., c:/16_Decision_Tree/Fraud_check.csv)
df = pd.read_csv("C:/16_Decision_Tree/Fraud_check.csv")

# ---------------------------------------------------------
# STEP 2: Target Discretization (As per Problem Statement)
# ---------------------------------------------------------
# Treat those who have taxable_income <= 30000 as Risky (1) and others as Good (0)
df['Target'] = np.where(df['Taxable.Income'] <= 30000, 'Risky', 'Good')

# Drop the original continuous column to prevent data leakage
df = df.drop(columns=['Taxable.Income'])

# ---------------------------------------------------------
# STEP 3: Categorical Encoding & Feature Engineering
# ---------------------------------------------------------
le_undergrad = LabelEncoder()
le_marital = LabelEncoder()
le_urban = LabelEncoder()
le_target = LabelEncoder()

df['Undergrad'] = le_undergrad.fit_transform(df['Undergrad'])
df['Marital.Status'] = le_marital.fit_transform(df['Marital.Status'])
df['Urban'] = le_urban.fit_transform(df['Urban'])
df['Target'] = le_target.fit_transform(df['Target']) # Good -> 0, Risky -> 1

# Define Features (X) and Target (y)
X = df.drop(columns=['Target'])
y = df['Target']

# Split data using stratify to keep proportions of Risky individuals intact
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# STEP 4: Feature Scaling (Requirement 5.1 - Scaled Data Options)
# ---------------------------------------------------------
# Scaling continuous metrics ('City.Population', 'Work.Experience')
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

num_cols = ['City.Population', 'Work.Experience']
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# ---------------------------------------------------------
# MODEL 1: DECISION TREE (BASELINE ON SCALED DATA)
# ---------------------------------------------------------
dt_baseline = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_baseline.fit(X_train_scaled, y_train)

y_train_pred_dt = dt_baseline.predict(X_train_scaled)
y_test_pred_dt  = dt_baseline.predict(X_test_scaled)

print("="*60)
print("BASELINE SCALED DECISION TREE PERFORMANCE")
print("="*60)
print("Decision Tree Train Accuracy :", accuracy_score(y_train, y_train_pred_dt))
print("Decision Tree Test Accuracy  :", accuracy_score(y_test, y_test_pred_dt))
print("\nConfusion Matrix (Test Data):\n", confusion_matrix(y_test, y_test_pred_dt))
print("\nClassification Report (Test Data):\n", classification_report(y_test, y_test_pred_dt))


# ----------------------------------------------------------------------------------------------
# DECISION TREE OPTIMIZATION (OVERFITTING CONTROL VIA CROSS-VALIDATION)
# ----------------------------------------------------------------------------------------------
dt_reg = DecisionTreeClassifier(criterion='entropy', random_state=42)

param_grid_dt = {
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [10, 20, 40],
    'min_samples_leaf': [5, 10, 20],
    'class_weight': ['balanced'] # Critical due to imbalanced fraud entries
}

grid_dt = GridSearchCV(estimator=dt_reg, param_grid=param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)
grid_dt.fit(X_train_scaled, y_train)

print("\nBest Parameters Found via CV:", grid_dt.best_params_)

dt_opt = grid_dt.best_estimator_
y_train_pred_opt = dt_opt.predict(X_train_scaled)
y_test_pred_opt  = dt_opt.predict(X_test_scaled)

print("\n" + "="*60)
print("OPTIMIZED DECISION TREE PERFORMANCE")
print("="*60)
print("Optimized DT Train Accuracy :", accuracy_score(y_train, y_train_pred_opt))
print("Optimized DT Test Accuracy  :", accuracy_score(y_test, y_test_pred_opt))
print("\nConfusion Matrix (Test Data):\n", confusion_matrix(y_test, y_test_pred_opt))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred_opt))


# ---------------------------------------------------------
# MODEL 2: RANDOM FOREST ENSEMBLE
# ---------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=10,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)

y_train_pred_rf = rf.predict(X_train_scaled)
y_test_pred_rf  = rf.predict(X_test_scaled)

print("\n" + "="*60)
print("RANDOM FOREST ENSEMBLE PERFORMANCE")
print("="*60)
print("Random Forest Train Accuracy:", accuracy_score(y_train, y_train_pred_rf))
print("Random Forest Test Accuracy :", accuracy_score(y_test, y_test_pred_rf))
print("\nConfusion Matrix (Test Data):\n", confusion_matrix(y_test, y_test_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred_rf))


# ----------------------------------------------------------------------------------------------
# MODEL METRIC COMPARISON TABLE GENERATION
# ----------------------------------------------------------------------------------------------
# Inverting or tracking target correctly: LabelEncoder mapped Good -> 0, Risky -> 1
metrics_compare = {
    'Model': ['Baseline Decision Tree', 'Optimized Decision Tree', 'Random Forest'],
    'Train Accuracy': [accuracy_score(y_train, y_train_pred_dt), accuracy_score(y_train, y_train_pred_opt), accuracy_score(y_train, y_train_pred_rf)],
    'Test Accuracy': [accuracy_score(y_test, y_test_pred_dt), accuracy_score(y_test, y_test_pred_opt), accuracy_score(y_test, y_test_pred_rf)],
    'Test Precision (Risky)': [precision_score(y_test, y_test_pred_dt, pos_label=1), precision_score(y_test, y_test_pred_opt, pos_label=1), precision_score(y_test, y_test_pred_rf, pos_label=1)],
    'Test Recall (Risky)': [recall_score(y_test, y_test_pred_dt, pos_label=1), recall_score(y_test, y_test_pred_opt, pos_label=1), recall_score(y_test, y_test_pred_rf, pos_label=1)]
}

df_compare = pd.DataFrame(metrics_compare)
print("\n" + "="*60)
print("FINAL MODEL PERFORMANCE COMPARISON")
print("="*60)
print(df_compare.to_string(index=False))

#=========================================================================
#Business Benefit & Strategic Impact Report
#=========================================================================
Deploying this automated predictive system introduces structural advantages to risk 
mitigation teams and financial auditors:

Optimized Auditing Allocations: Instead of wasting manual billable hours running 
random field investigations on low-risk accounts, audit teams can now focus directly
 on profiles flagged as Risky by the algorithm.

Proactive Tax Fraud Containment: By screening customer demographics (Undergrad, 
Marital.Status, City.Population, Work.Experience) concurrently, the system spots
 early operational risk footprints, helping prevent evasion before files are 
 processed for validation.

Minimizing Defect Escape: Moving from single decision lines to a 300-tree Random
 Forest prevents overfitting, ensuring the institution stays protected against new
 and evolving tax evasion strategies.