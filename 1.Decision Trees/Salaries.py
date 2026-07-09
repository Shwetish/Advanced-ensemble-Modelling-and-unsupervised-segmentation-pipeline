# ------------------------------------------------------------------------------
# BUSINESS UNDERSTANDING
# ---------------------------------------------------------------------
#1. Business Problem Statement
#When tech companies look to scale their workforce, human resource teams
# struggle to offer competitive compensation packages that accurately match 
#an applicant's specific profile. Hiring teams do not have an automated, 
#structured way to immediately pinpoint which talent combinations—based on 
#target employer background, core technical job roles, and academic 
#degrees—deserve a premium high-tier salary over a baseline offer.

#2. Business Objective
#To build a clear classification rule engine that predicts whether a 
#professional's profile will command a high-tier baseline salary of over
# $100,000 based on three core inputs: their company, their specific job 
#role, and their highest degree.

#3. Motivation
#Competitive Talent Sourcing: Instantly recognize what specific technical or
# management profiles command top-tier compensation in the current job market
# to win competitive talent battles.
#Standardized Offer Letters: Eliminate slow, manual salary guessing games 
#during late-stage negotiations, allowing recruiters to make data-backed
# offers instantly.

#Optimized Operational Budgets: Keep hiring spending under control by 
#identifying which specific feature mixes do not require a high-tier 
#compensation premium.

#4. Constraints
#Categorical Feature Dominance: The dataset is composed entirely of 
#text-based categorical columns (company, job, degree) rather than continuous
# numbers. These must be cleanly encoded before building a model.

#Extremely Small Profile Count: The sample footprint is short and concise
# (fewer than 20 rows visible). This makes models highly vulnerable to 
#overfitting or simply memorizing the exact rows instead of learning a 
#pattern.

#5. Business Success Criteria
#Equip recruiters with an objective tool that correctly aligns compensation
# structures with candidate backgrounds during intake screenings.
#Accelerate the speed of generating formal job offers for high-demand roles 
#like computer programmer or business manager.

#6. ML Success Criteria
#Perfect Model Interpretability: Because the criteria rely on specific paths
# (e.g., matching a Master's degree at Google vs. Facebook), the Decision 
#Tree must produce clear, logical rules that human HR staff can read and 
#trust.

#Achieve stable prediction rules through careful pruning (controlling tree 
#depth), preventing a single small training record from distorting the 
#overall logic.

#=================================================================================
#Data Understanding:-
#================================================================================
'''
Name of Feature             Description                    Type          Relevance
company         The tech company employer name       .Qualitative,      Highly
                (google, abc pharma, or facebook)     Nominal           Relevant
jobThe          candidate's specific job role
                (sales executive, business manager,  Qualitative,       Highly
                 or computer programmer).            Nominal            Relevant
degree          The highest education qualification
                 earned by the worker                Qualitative,      Partially
                 (bachelors or masters).             Nominal           Relevant
salary_more_     Shows if the person earns
than_100k        more than $\$100,000$ a year       Qualitative,       Target
                 (1 for Yes, 0 for No).             Nominal(Binary)    Variable
'''
# ==============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# Salaries Dataset - Role-Based Salary Premium Benchmarking
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("c:/16_Decision_Tree/salaries.csv")

df.head()

# Internal categorical encoding to perform distribution moment calculations
eda_df = df.copy()
le = LabelEncoder()
for col in ['company', 'job', 'degree']:
    eda_df[col] = le.fit_transform(eda_df[col])

#--------------------------------------------------------------------------
# 1 First Moment - Mean
#--------------------------------------------------------------------------
eda_df.mean(numeric_only=True)
'''
Inference:
Mean represents the baseline distribution center of the encoded corporate profiles.

salary_more_than_100k Mean -> Represents the proportion of premium earners (~62.5%).

Used for:
Establishing the baseline premium salary probability across the surveyed organizations.
'''

#----------------------------------------------------------------------------
# 2 Second Moment - Variance & Standard Deviation
#----------------------------------------------------------------------------
eda_df.var(numeric_only=True)
eda_df.std(numeric_only=True)
'''
Inference:
Moderate variance observed in:
- company
- job

Business Meaning:
The dataset contains a balanced, structured mix of corporate entities (Google, ABC Pharma, Facebook) 
and structural job designations, ensuring no single role type dominates the sample background.
'''

#-----------------------------------------------------------------------------
# 3 Third Moment - Skewness
#-----------------------------------------------------------------------------
eda_df.skew(numeric_only=True)
'''
Feature                    Skewness         Business Insight
---                        ---              ---
company                    Near symmetric   Balanced representation across firms.
job                        Near symmetric   Even layout of job roles in the dataset.
degree                     Near symmetric   Even sampling of Bachelors vs Masters levels.
salary_more_than_100k      Left skewed      The dataset skews toward positions making >100k.

Inference:
The categorical inputs show structural symmetry due to uniform profile tracking.
Tree-based models process these distinct groupings easily without needing forced shifts.
'''

#------------------------------------------------------------------------------
# 4 Fourth Moment - Kurtosis
#------------------------------------------------------------------------------
eda_df.kurtosis(numeric_only=True)
'''
Inference:
company (-1.55) / job (-1.53) / degree (-2.26)
Highly platykurtic (flat distribution peaks), reflecting uniform sampling 
across structural categorical choices without sharp clusters.

salary_more_than_100k (-1.77)
Platykurtic distribution, indicating a clear, binary distribution between the two earnings brackets.
'''

#----------------------------------------------------------------------------------------------
# Step 8: Histograms (Distribution Analysis)
#----------------------------------------------------------------------------------------------
eda_df.hist(figsize=(12,8), edgecolor='black')
plt.suptitle("Histogram of Encoded Categorical Attributes and Target")
plt.show()
'''
Inference:
company / job / degree:
Display uniform categorical step blocks, proving a balanced matrix layout.

salary_more_than_100k:
Shows a clear binary count, confirming higher representation in the premium group (>100k).
''' 

#----------------------------------------------------------------------------------------------   
# Step 9: Boxplots (Outlier Detection)
#----------------------------------------------------------------------------------------------
plt.figure(figsize=(12,5))
sns.boxplot(data=eda_df, orient='h')
plt.title("Boxplot of All Features")
plt.show()
'''
Inference:
company / job / degree / salary_more_than_100k:
Zero statistical outliers exist in this dataset.

Business Interpretation:
Since the parameters are fully categorical and perfectly balanced, there are no outlier anomalies.
Tree-based models can build clean, unskewed decision paths.
'''

#---------------------------------------------------------------------------------------------
# Step 10: Target Variable Distribution (Salary Premium Category)
#--------------------------------------------------------------------------------------------
sns.countplot(x='salary_more_than_100k', data=df)
plt.title("Salary Premium Category Distribution (0: <=100k, 1: >100k)")
plt.show()
'''
Inference:
The target distributions are highly balanced:
>100k (1)  → Majority Class (~62.5%)
<=100k (0) → Minority Class (~37.5%)

Severe class imbalance is completely absent. SMOTE is NOT required.

Why SMOTE is a bad idea here:
Tree-based models evaluate pure categorical branch groupings perfectly at this ratio.
Using SMOTE would generate duplicate categorical combinations, risking overfitting.

Best practice for this dataset:
Train the models directly using standard criterion tracking without synthetic modifications.
'''

#----------------------------------------------------------------------------------------------
# Step 11: Correlation Heatmap
#----------------------------------------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.heatmap(eda_df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
'''
company vs salary_more_than_100k (0.19)
Weak positive correlation; suggests certain organizations lean slightly higher in premium distribution.

job vs salary_more_than_100k (-0.21)
Weak inverse correlation; indicates designation types heavily guide salary outcomes.

degree vs salary_more_than_100k (0.26)
Maintains the strongest linear tracking to salary, highlighting education as a key driver.
'''

#------------------------------------------------------------------------------
# Step 12: Scatter Plot (Business Relationship)
#------------------------------------------------------------------------------
sns.stripplot(x='job', y='company', hue='salary_more_than_100k', data=df, jitter=True, size=10)
plt.title("Company vs Job Designation Earning Matrix")
plt.show()
'''
Inference:
Visually maps out premium salary concentrations across specific job-firm intersections (e.g., Facebook roles).
'''

#------------------------------------------------------------------------------
# Step 13: PDF & CDF Analysis
#------------------------------------------------------------------------------
for col in eda_df.columns:
    plt.figure(figsize=(12,3))

    # PDF
    plt.subplot(1,2,1)
    sns.kdeplot(eda_df[col], fill=True)
    plt.title(f'PDF of {col}')

    # CDF 
    plt.subplot(1,2,2)
    sorted_vals = np.sort(eda_df[col])
    y_vals = np.arange(len(sorted_vals)) / len(sorted_vals)
    plt.plot(sorted_vals, y_vals)
    plt.title(f'CDF of {col}')
    plt.show()
'''
Inference:
PDF shows distinct categorical density steps rather than continuous curves.
CDF helps track cumulative categorical probability thresholds.
'''

#------------------------------------------------------------------------------
# Step 14: Pairplot (Feature Interaction)
#------------------------------------------------------------------------------
sns.pairplot(eda_df, hue='salary_more_than_100k')
plt.show()
'''
Inference:
Highlights clear categorical cross-sections where premium compensation models apply.

# Step 15: Final EDA Summary
- Dataset is clean, complete, and contains no missing entries.
- Features are entirely categorical and evenly distributed.
- No outliers are present, making the dataset highly stable.

Suitable Models:
Decision Tree Classifier
Random Forest Classifier
'''

# ==============================================================================
# DATA PREPROCESSING & MODEL BUILDING ON SALARIES DATA
# ==============================================================================

# ---------------------------------------------------------
# Step 1: Import Required Libraries
# ---------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set(style="whitegrid")

#----------------------------------------------------------------------------
# Step 2: Load Dataset
#----------------------------------------------------------------------------
df = pd.read_csv("c:/16_Decision_Tree/salaries.csv")

print("Initial Shape:", df.shape)
df.head()
'''
Inference:
Dataset loaded successfully with categorical profile indices.
'''

#--------------------------------------------------------------------------
# Step 3: Data Type Check
#--------------------------------------------------------------------------
df.dtypes
'''
Inference:
Features are object types (strings). Label encoding is required.
'''

#------------------------------------------------------------------------
# Step 4: Missing Value Analysis
#-------------------------------------------------------------------------    
print("Missing values:\n", df.isnull().sum())
'''
Inference:
No missing values detected.
'''

#---------------------------------------------------------
# Step 5: Duplicate Removal
#---------------------------------------------------------
# Note: In an ultra-small categorical demographic dataset (16 rows), identical combinations 
# represent valid multiple occurrences of that job type rather than systematic duplicates.
# Therefore, standard deduplication is skipped to protect dataset volume.
print("Retaining corporate matrix shape:", df.shape)

#---------------------------------------------------------
# Step 6: Target Variable Pre-Check
#---------------------------------------------------------
print(df['salary_more_than_100k'].value_counts())
'''
Inference:
Distribution is well-balanced. No sampling adjustments required.
'''

#--------------------------------------------------------------------------------
# Step 7: Encoding Categorical Variables
#-------------------------------------------------------------------------------
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

df['company'] = le_company.fit_transform(df['company'])
df['job']     = le_job.fit_transform(df['job'])
df['degree']  = le_degree.fit_transform(df['degree'])

'''
Why Label Encoding?
Tree-based models split categorical levels cleanly by index position.
This avoids high-dimensional column expansion.
'''

# ---------------------------------------------------------
# Step 8: Outlier Detection (Boxplots)
# ---------------------------------------------------------
# Outlier step omitted because data features are entirely discrete categorical values.

# ---------------------------------------------------------
# Step 9: Outlier Treatment (Winsorization)
# ---------------------------------------------------------
# Outlier treatment omitted as no extreme values exist.

# ---------------------------------------------------------
# Step 10: Skewness Detection
# ---------------------------------------------------------
# Skewness transformations are omitted for tree models on encoded categorical columns.

# ---------------------------------------------------------
# Step 11: Feature Scaling
# ---------------------------------------------------------
'''
Feature scaling is NOT required for:
Decision Tree / Random Forest
Reason:
Tree models split nodes based on category purity values, making them scale-invariant.
'''

# ---------------------------------------------------------
# Step 12: Train-Test Split Setup
# ---------------------------------------------------------
X = df.drop(columns=['salary_more_than_100k'])
y = df['salary_more_than_100k']

# Small dataset split using stratification to guarantee representative test splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train Size: {X_train.shape} | Test Size: {X_test.shape}")

# ---------------------------------------------------------
# Step 13: Build and Evaluate Decision Tree Model
# ---------------------------------------------------------
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
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
rf_model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("\n" + "="*50)
print("RANDOM FOREST EVALUATION METRICS")
print("="*50)
print("Accuracy Score:", accuracy_score(y_test, rf_preds))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, rf_preds))
print("\nClassification Report:\n", classification_report(y_test, rf_preds))

# ---------------------------------------------------------
# MODEL 1: DECISION TREE (BASELINE)
# ---------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize Baseline Decision Tree model
dt = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
)

# Train the model
dt.fit(X_train, y_train)

# Predictions on training and testing data
y_train_pred = dt.predict(X_train)
y_test_pred  = dt.predict(X_test)

# Model performance
print("Decision Tree Train Accuracy :", accuracy_score(y_train, y_train_pred))
print("Decision Tree Test Accuracy  :", accuracy_score(y_test, y_test_pred))

# Confusion Matrix
print("\nConfusion Matrix (Test Data):\n")
print(confusion_matrix(y_test, y_test_pred))

# Classification Report
print("\nClassification Report (Test Data):\n")
print(classification_report(y_test, y_test_pred))

#----------------------------------------------------------------------------------------------
# DECISION TREE OPTIMIZATION (OVERFITTING CONTROL)
#----------------------------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# STEP 1: Define a Regularized Decision Tree
dt_reg = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
)

# STEP 2: Hyperparameter Grid (Controls Complexity)
# Adjusted to respect the compact size of the salaries dataset matrix
param_grid = {
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3]
}

'''
Why these parameters?
max_depth         -> Limits tree structural growth to prevent memorization.
min_samples_split -> Ensures nodes only break when sufficient data supports the split.
min_samples_leaf  -> Clamps leaf sizes to protect against noise isolation.
'''

#----------------------------------------------------------------------------------------------
# STEP 3: GridSearchCV (Cross-Validated Optimization)
#-------------------------------------------------------------------------------------------------
grid_dt = GridSearchCV(
    estimator=dt_reg,
    param_grid=param_grid,
    cv=3,  # Adjusted to match the smaller corporate matrix footprint
    scoring='accuracy',
    n_jobs=-1
)

grid_dt.fit(X_train, y_train)

# STEP 4: Best Model from Grid Search
print("Best Parameters:", grid_dt.best_params_)

# STEP 5: Train Optimized Decision Tree
dt_opt = grid_dt.best_estimator_

# Predictions
y_train_pred_opt = dt_opt.predict(X_train)
y_test_pred_opt = dt_opt.predict(X_test)

# Performance
print("Optimized DT Train Accuracy :", accuracy_score(y_train, y_train_pred_opt))
print("Optimized DT Test Accuracy  :", accuracy_score(y_test, y_test_pred_opt))

# Confusion Matrix
print("\nConfusion Matrix (Test Data):\n")
print(confusion_matrix(y_test, y_test_pred_opt))

# Pass zero_division=0 to silence the warnings cleanly
print("\nClassification Report")
print(classification_report(y_test, y_test_pred_opt, zero_division=0))

'''
Inference:
The baseline Decision Tree naturally attempts to perfectly isolate every corporate permutation.
By applying GridSearch pruning, we force the tree to learn broader, row-independent patterns.
However, a single tree remains structurally vulnerable to variance on small categorical profiles.

WHY THIS IS HAPPENING (Very Important)
1 Dataset Characteristics:
  Small categorical matrices present high density blocks where single variable changes swap the target state completely.

2 Decision Trees are HIGH-VARIANCE MODELS:
  Even with max_depth and structural limitations applied, a single decision boundary tree:
  - Tends to memorize rigid cross-sectional combinations.
  - Changes its node topology significantly with small shifts in train rows.
  This represents a standard mathematical constraint, not a tuning error.

3 Metric Context:
  For a binary profile dataset with premium compensation leaning at 62.5%, accuracy must be balanced 
  against macro precision and recall parameters to ensure low earners are not systematically misclassified.

WHAT IS THE CORRECT NEXT STEP?
You do NOT fight overfitting further with a single Decision Tree.
You change the model class to an ensemble structure.
'''

#----------------------------------------------------------------------------------------------
# RANDOM FOREST ENSEMBLE CLASS
#----------------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest to aggregate multiple uncorrelated decision paths
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_train_pred_rf = rf.predict(X_train)
y_test_pred_rf = rf.predict(X_test)

print("Random Forest Train Accuracy:", accuracy_score(y_train, y_train_pred_rf))
print("Random Forest Test Accuracy :", accuracy_score(y_test, y_test_pred_rf))

'''
Inference:
This is exactly the right moment to pause and interpret, not panic.
What you're seeing now is NOT a tuning failure — it's a data-limited ceiling.

By bootstrapping samples and sub-selecting features at each node split, Random Forest 
stabilizes the predictions, protecting the model from local noise and providing a realistic 
look at true generalization.
'''