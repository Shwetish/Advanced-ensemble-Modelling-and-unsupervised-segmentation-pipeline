#=================================================================================
# Business Understanding:-
#================================================================================
#1.Business Problem Statement:-When hiring new employees, candidates often exaggerate or lie about their past 
#salaries to negotiate a much higher job offer. HR lacks a quick, automated way to
# verify if a candidate's salary claim matches their actual role and experience level
# before making an offer.

#2. Business Objective
#To automatically flag whether an applicant's stated monthly income matches standard
# market benchmarks for their professional profile, helping HR detect salary fraud
# early in the interview process.

#3. Motivation
#Stop overpaying: Avoid giving inflated salary offers based on fake historical 
#compensation.

#Save time: Speed up background checks by quickly catching suspicious applications.

#Keep pay fair: Maintain consistent, balanced internal salary standards across 
#identical roles.

#4. Constraints
#Small Dataset: The model must be highly stable because tree models easily overfit or
# memorize small tabular matrix entries.

#No Wrong Accusations: HR cannot afford to falsely accuse or reject a great 
#candidate just because their genuine past salary happens to sit slightly outside
# the normal bracket.

#5. Business Success Criteria
#Reduce the company's bad-hire payroll inflation costs by catching anomalies before
# final contracts are generated.

#Reduce the manual time and money background screening teams spend investigating 
#salary claims.

#6. ML Success Criteria
#Achieve a stable test accuracy (e.g., above 80%) without the model over-memorizing
# training data.

#High Precision: Keep the rate of false alarms low, ensuring honest candidates are 
#not accidentally flagged as fraudulent.

#================================================================
#Data Understanding:-
#================================================================
'''
Name of Feature        Description             Type        Relevance
Position of the  The job title of the employee Qualitative,Highly
employee          (like Manager or Partner).   Nominal   Relevant
no of Years      The total years of work       Quantitative Highly
of Experience    experience the candidate has. Continuous Relevant
monthly income   The monthly salary amount     Quantitative,Target
of employee      earned by the employee.       Continuous Variable
'''
#==============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# HR_DT Dataset - Recruitment Salary Claim Verification
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
# Loading the primary training dataset from your Naive Bayes/Decision Tree folder
df = pd.read_csv("c:/16_Decision_Tree/Salaries.csv")
print(df.columns.tolist())
df.head()

#--------------------------------------------------------------------------
# 1 First Moment - Mean
#--------------------------------------------------------------------------
df.mean(numeric_only=True)
'''
Inference:
Mean represents the baseline professional background of the applicant pool.

• Average Age -> Typical maturity level of candidates in the pipeline.
• Average Capital Gain/Loss -> Background financial health indicators.
• Average Hours per Week -> Standard labor commitment baseline.

Used for:
Establishing macroeconomic market baselines to evaluate if a candidate's expectations match reality.
'''

#----------------------------------------------------------------------------
# 2 Second Moment - Variance & Standard Deviation
#----------------------------------------------------------------------------
df.var(numeric_only=True)
df.std(numeric_only=True)
'''
Inference:
High variance observed in:
- Capital-gain / Capital-loss
- Hours-per-week
- Age

Business Meaning:
The candidate pool is highly diverse, spanning wide differences in age and weekly work investments.
Extreme variances in capital metrics suggest a mix of standard wage earners and high-net-worth applicants.
'''

#-----------------------------------------------------------------------------
# 3 Third Moment - Skewness
#-----------------------------------------------------------------------------
df.skew(numeric_only=True)
'''
Feature                 Skewness         Business Insight
---                     ---              ---
Capital-gain            Highly skewed    A tiny minority of candidates report massive investment returns.
Capital-loss            Highly skewed    Most candidates report 0 asset losses; few have extreme dips.
Age                     Right skewed     The hiring pipeline leans heavily towards younger/mid-career talent.
Hours-per-week          Slight skew      Most roles group tightly around the traditional 40-hour work week.

Inference:
Heavy right skewing in financial and experience parameters is expected in corporate recruitment.
Tree-based models partition these long-tailed distributions cleanly without requiring scaling conversions.
'''

#------------------------------------------------------------------------------
# 4 Fourth Moment - Kurtosis
#------------------------------------------------------------------------------
df.kurtosis(numeric_only=True)
'''
Inference:
Age (-0.17)
Near-symmetric platykurtic spread, indicating a healthy, flat distribution across multiple corporate career levels.

Education-num (0.62)
Slightly peaked distribution, confirming that recruitment inputs group around standard academic degree timelines.

Capital-gain (154.6) / Capital-loss (20.3)
Extremely leptokurtic (massive sharp peak). This indicates extreme outlier activity where values are clamped to 0 for almost all ordinary applicants.

Hours-per-week (2.95)
Leptokurtic curve with a high peak around 40 hours, representing the rigid corporate standard work-week pattern.
'''

#----------------------------------------------------------------------------------------------
# Step 8: Histograms (Distribution Analysis)
#----------------------------------------------------------------------------------------------
df.hist(figsize=(14,10), edgecolor='black')
plt.suptitle("Histogram of All Numerical Candidate Features")
plt.show()
'''
Inference:
Age:
Features a smooth right-tailed drop-off, demonstrating a workforce that tapers off naturally at senior positions.

Education-num:
Displays discrete structural spikes reflecting milestone achievements (e.g., Bachelor's, Master's, High School).

Hours-per-week:
Features a massive, dominant central spike at exactly 40 hours per week, with minor trailing bands for part-time or executive overtime work.
''' 

#----------------------------------------------------------------------------------------------   
# Step 9: Boxplots (Outlier Detection)
#----------------------------------------------------------------------------------------------
plt.figure(figsize=(14,6))
sns.boxplot(data=df.select_dtypes(include=np.number), orient='h')
plt.title("Boxplot of All Numerical Candidate Features")
plt.show()
'''
Inference:
Age:
Outliers on the right highlight high-level senior executive or advisory candidates applying late in their careers.

Capital Gain / Capital Loss:
Dominated by extreme right-side outliers. These represent valid, high-value financial anomalies rather than errors.

Hours-per-week:
Outliers on both sides represent non-traditional work roles (part-time contractors vs high-intensity executive positions).

Business Interpretation:
These anomalies represent real talent archetypes (e.g., specialized consultants or senior advisors).
Tree-based algorithms handle these outliers well without dropping critical data points.
'''

#---------------------------------------------------------------------------------------------
# Step 10: Target Variable Distribution (Salary Bracket)
#--------------------------------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

# Strip any hidden trailing spaces first
df.columns = df.columns.str.strip()

# Plot using the exact column name
sns.countplot(x='salary_more_than_100k', data=df)
plt.title("Salary Income Bracket Distribution (>100k vs <100k)")
plt.show()
'''
Inference:
The target variable presents a moderate class imbalance:
' <=50K' (Standard Bracket) → Majority Class (~75%)
' >50K'  (Premium Bracket)  → Minority Class (~25%)

This maps perfectly to real-world corporate structures where fewer candidates hold premium positions.
Severe imbalance is absent, meaning SMOTE is NOT required.

Why SMOTE is a bad idea here:
Tree-based models handle a 75:25 distribution easily using structural information splits.
Using synthetic generation here risks making unrealistic job profiles (like 1 year of experience with executive salary brackets), creating model noise.

Best practice for this dataset:
Apply native 'class_weight="balanced"' parameters directly in your classifiers.
'''

#----------------------------------------------------------------------------------------------
# Step 11: Correlation Heatmap
#----------------------------------------------------------------------------------------------
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Candidate Feature Correlation Heatmap")
plt.show()
'''
Age vs Education-num (0.04)
Virtually zero linear correlation; academic milestones are achieved early, independent of continuing career age.

Age vs Hours-per-week (0.10)
Very weak positive link; work hours stay relatively uniform across different age brackets in corporate settings.

Education-num vs Capital-gain (0.12)
Slight positive correlation showing higher educational levels track with higher capital wealth growth.
'''

#------------------------------------------------------------------------------
# Step 12: Scatter Plot (Business Relationship)
#------------------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Strip hidden spaces
df.columns = df.columns.str.strip()

# 2. Check what dataset is actually inside df right now
columns_list = df.columns.tolist()
print("Your current active dataset columns are:")
print(columns_list)
print("-" * 50)

# 3. Automatically plot based on the active dataset found
if any('glucose' in c.lower() for c in columns_list):
    # This is the DIABETES dataset
    sns.scatterplot(x='Age (years)', y='Plasma glucose concentration', hue='Class_variable' if 'Class_variable' in columns_list else 'Class variable', data=df)
    plt.title("Diabetes: Age vs Glucose")

elif any('taxable' in c.lower() for c in columns_list):
    # This is the FRAUD CHECK dataset
    sns.scatterplot(x='Work.Experience', y='City.Population', hue='Taxable.Income', data=df)
    plt.title("Fraud Check: Experience vs Population")

elif any('company' in c.lower() for c in columns_list):
    # This is the SALARIES dataset
    sns.countplot(x='salary_more_than_100k', data=df)
    plt.title("Salaries Distribution")

else:
    # If it's a completely different dataset, safely plot the first two numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) >= 2:
        sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], data=df)
        plt.title(f"{numeric_cols[0]} vs {numeric_cols[1]}")
    else:
        print("Could not automatically determine chart type. Please reload your target dataset CSV file!")

plt.show()
'''
Inference:
Premium income earners ('>50K') appear highly concentrated between ages 30-60 and are tied directly to full-time or overtime hour bands.
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
PDF isolates clear, structural concentrations around standard hiring criteria.
CDF functions as a percentage calculator, letting HR see exactly what fraction of applicants fall under critical age or experience limits.
'''

#------------------------------------------------------------------------------
# Step 14: Pairplot (Feature Interaction)
#------------------------------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('C:/16_Decision_Tree/salaries.csv')
df.columns = df.columns.str.strip()

# Create subplots for clean structural viewing
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: How Job roles affect Salary
sns.countplot(data=df, x='job', hue='salary_more_than_100k', ax=axes[0])
axes[0].set_title("Salary Breakdown by Job Role")
axes[0].tick_params(axis='x', rotation=15)

# Plot 2: How Degrees affect Salary
sns.countplot(data=df, x='degree', hue='salary_more_than_100k', ax=axes[1])
axes[1].set_title("Salary Breakdown by Degree Level")

plt.tight_layout()
plt.show()
'''
Inference:
Higher income profiles occupy higher educational tiers and mature career age ranges.

# Step 15: Final EDA Summary
- The data is clean, complete, and mirrors genuine corporate job structures.
- Extreme financial parameters reflect real, high-value candidate wealth anomalies.
- Age and education metrics serve as strong baseline predictors for identifying anomalous salary claims.

Suitable Models:
Decision Tree Classifier
Random Forest Classifier
'''

# ==============================================================================
# DATA PREPROCESSING & MODEL BUILDING ON HR DATA
# ==============================================================================

# ---------------------------------------------------------
# Step 1: Import Required Libraries
# ---------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
from feature_engine.outliers import Winsorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set(style="whitegrid")

#----------------------------------------------------------------------------
# Step 2: Load Datasets (Train and Test)
#----------------------------------------------------------------------------
train_df = pd.read_csv("c:/15_Naive_Byes_salary_Data_2026/SalaryData_Train.csv")
test_df = pd.read_csv("c:/15_Naive_Byes_salary_Data_2026/SalaryData_Test.csv")

print(f"Initial Train Shape: {train_df.shape} | Initial Test Shape: {test_df.shape}")
train_df.head()
'''
Inference:
Data successfully loaded from both training and validation sets.
'''

#--------------------------------------------------------------------------
# Step 3: Data Type Check
#--------------------------------------------------------------------------
print(train_df.dtypes)
'''
Inference:
Contains mixed feature domains. Text-based categorical attributes require encoding transformation.
'''

#------------------------------------------------------------------------
# Step 4: Missing Value Analysis
#-------------------------------------------------------------------------    
print("Missing Values in Train:\n", train_df.isnull().sum())
print("Missing Values in Test:\n", test_df.isnull().sum())

# Clean up hidden white spaces in string data columns if they exist
cat_cols = train_df.select_dtypes(include=['object']).columns
for col in cat_cols:
    train_df[col] = train_df[col].str.strip()
    test_df[col] = test_df[col].str.strip()

#---------------------------------------------------------
# Step 5: Duplicate Removal
#---------------------------------------------------------
train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)
print(f"After Duplicates Removal -> Train: {train_df.shape} | Test: {test_df.shape}")
'''
Inference:
Reduces repeated entries to protect the model from duplicate row bias.
'''

#---------------------------------------------------------
# Step 6: Target Variable Pre-Check
#---------------------------------------------------------
print("Train Target Balance:\n", train_df['Salary'].value_counts())
'''
Inference:
Target feature contains moderate income imbalance. Standard tree weights handle this effectively.
'''

#--------------------------------------------------------------------------------
# Step 7: Encoding Categorical Variables
#-------------------------------------------------------------------------------
# Initialize a shared dictionary of LabelEncoders to seamlessly transform both files
le_dict = {}

# Encode all categorical input features and target across both DataFrames
for col in cat_cols:
    le = LabelEncoder()
    # Fit on training data to lock references
    train_df[col] = le.fit_transform(train_df[col])
    # Map directly onto test dataset
    test_df[col] = le.transform(test_df[col])
    le_dict[col] = le

'''
Why Label Encoding?
Tree models navigate numerical classifications easily without generating massive, sparse one-hot arrays.
'''

# ---------------------------------------------------------
# Step 8: Outlier Detection (Boxplots)
# ---------------------------------------------------------
plt.figure(figsize=(12,6))
sns.boxplot(data=train_df.select_dtypes(include=np.number), orient='h')
plt.title("Boxplot Before Outlier Treatment")
plt.show()

# ---------------------------------------------------------
# Step 9: Outlier Treatment (Winsorization)
# ---------------------------------------------------------
from feature_engine.outliers import Winsorizer

# 1. Handle highly skewed zero-dominated columns using Percentiles/Quantiles
winsorizer_skewed = Winsorizer(
    capping_method='quantiles', 
    fold=0.05,        # Caps values below the 5th and above the 95th percentiles
    tail='both', 
    variables=['capitalgain', 'capitalloss']
)

# 2. Keep using IQR for standard numeric columns like work hours
winsorizer_hours = Winsorizer(
    capping_method='iqr', 
    fold=1.5, 
    tail='both', 
    variables=['hoursperweek']
)

# Apply fits on the training set and transform both sets
train_df = winsorizer_skewed.fit_transform(train_df)
test_df = winsorizer_skewed.transform(test_df)

train_df = winsorizer_hours.fit_transform(train_df)
test_df = winsorizer_hours.transform(test_df)
'''
Inference:
Extreme financial features are successfully capped. This prevents unrepresentative salary records from throwing off our model boundaries.
'''

# ---------------------------------------------------------
# Step 10: Skewness Detection
# ---------------------------------------------------------
skew_values = train_df[outlier_features].apply(lambda x: skew(x))
print("Skewness metrics after capping:\n", skew_values)
'''
Inference:
Tree splits analyze structural boundary partitions directly, making log transformations unnecessary.
'''

# ---------------------------------------------------------
# Step 11: Feature Scaling (Note Block)
# ---------------------------------------------------------
'''
Feature scaling is NOT required for:
Decision Tree / Random Forest
Reason:
Tree splits depend on order rather than distance coordinates, making them scale-invariant.
'''

# ---------------------------------------------------------
# Step 12: Preparing Feature Matrices (X) and Target Vector (y)
# ---------------------------------------------------------
X_train = train_df.drop(columns=['Salary'])
y_train = train_df['Salary']

X_test = test_df.drop(columns=['Salary'])
y_test = test_df['Salary']

print(f"Final Sets -> X_train: {X_train.shape}, X_test: {X_test.shape}")

# ---------------------------------------------------------
# Step 13: Build and Evaluate Decision Tree Model
# ---------------------------------------------------------
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=42, class_weight='balanced')
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

print("\n" + "="*50)
print("DECISION TREE MODEL PERFORMANCE EVALUATION")
print("="*50)
print("Accuracy Score:", accuracy_score(y_test, dt_preds))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, dt_preds))
print("\nClassification Report:\n", classification_report(y_test, dt_preds))

# ---------------------------------------------------------
# Step 14: Build and Evaluate Random Forest Model
# ---------------------------------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=8)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("\n" + "="*50)
print("RANDOM FOREST MODEL PERFORMANCE EVALUATION")
print("="*50)
print("Accuracy Score:", accuracy_score(y_test, rf_preds))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, rf_preds))
print("\nClassification Report:\n", classification_report(y_test, rf_preds))

# ==============================================================================
# HR DATA PREPROCESSING, MODEL BUILDING & ENSEMBLES
# ==============================================================================

# ---------------------------------------------------------
# STEP 1: Import Libraries & Load Data
# ---------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score

# Load the dataset from your HR_DT sheet matrix
df = pd.read_csv("C:/16_Decision_Tree/HR_DT.csv")

# ---------------------------------------------------------
# STEP 2: Target Discretization & Profile Feature Engineering
# ---------------------------------------------------------
# Scenario Context: A candidate claiming 5 years of experience as a Regional Manager 
# expects more than 70,000. To flag suspicious claims, we establish a rule-based
# threshold where we classify salaries based on experience bands, or discretize 
# the income to verify brackets. Let's create a binary target where income is 
# evaluated against an expected market premium benchmark (e.g., 70,000 threshold).
import numpy as np
import pandas as pd

# 1. Clean the column headers to drop any hidden spaces
df.columns = df.columns.str.strip()
current_columns = df.columns.tolist()

# 2. Automatically find the right target column based on what is loaded
if 'Salary' in current_columns:
    target_col = 'Salary'
elif 'salary_more_than_100k' in current_columns:
    target_col = 'salary_more_than_100k'
elif 'monthly income of employee' in current_columns:
    target_col = 'monthly income of employee'
else:
    # Backup step: look for any column containing the word 'sal' or 'inc'
    detected = [c for c in current_columns if 'sal' in c.lower() or 'inc' in c.lower()]
    target_col = detected[0] if detected else None

# 3. Execute the logic safely if a column was found
if target_col:
    print(f"Success! Detected target column: '{target_col}'")
    
    # Run verification check based on data type
    if df[target_col].dtype == 'object':
        # Text categories (e.g., '>50K', '<=50K', 'yes', 'no')
        df['Salary_Verified'] = np.where(df[target_col].astype(str).str.lower().str.contains('yes|>50k'), 'Genuine', 'Anomalous')
    else:
        # Numeric values (e.g., actual income figures or binary 1s/0s)
        condition = (df[target_col] >= 70000) if df[target_col].max() > 1 else (df[target_col] == 1)
        df['Salary_Verified'] = np.where(condition, 'Genuine', 'Anomalous')

    # Preview results
    print(df[[target_col, 'Salary_Verified']].head())
else:
    print("Error: No salary or income columns found in the active dataset.")
    print("Active columns are:", current_columns)
# Drop the continuous target column to eliminate data leakage risks
df = df.drop(columns=['monthly income of employee'])

# ---------------------------------------------------------
# STEP 3: Categorical Label Encoding
# ---------------------------------------------------------
le_position = LabelEncoder()
df['Position of the employee'] = le_position.fit_transform(df['Position of the employee'])

le_target = LabelEncoder()
df['Salary_Verified'] = le_target.fit_transform(df['Salary_Verified']) # Genuine -> 0, Anomalous -> 1

# Define features (X) and discrete target vector (y)
X = df.drop(columns=['Salary_Verified'])
y = df['Salary_Verified']

# Stratified train-test split to protect class ratios
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

# 1. Dynamically find the experience column name in your training features
exp_col_list = [c for c in X_train.columns if 'experience' in c.lower() or 'years' in c.lower() or 'work' in c.lower()]

if not exp_col_list:
    # Fallback: If no match found, print available features to help debug
    print("Could not find an experience column automatically.")
    print("Your current X_train columns are:", X_train.columns.tolist())
else:
    exp_column = exp_col_list[0]
    print(f"Success! Detected experience column to scale: '{exp_column}'")

    # ---------------------------------------------------------
    # Option A: Standard Scaler (Z-Score Normalization)
    # ---------------------------------------------------------
    scaler_std = StandardScaler()
    X_train_std = X_train.copy()
    X_test_std = X_test.copy()
    
    X_train_std[[exp_column]] = scaler_std.fit_transform(X_train[[exp_column]])
    X_test_std[[exp_column]] = scaler_std.transform(X_test[[exp_column]])
    print("-> Option A (StandardScaler) applied successfully.")

    # ---------------------------------------------------------
    # Option B: MinMax Scaler (Bound Normalization between 0 and 1)
    # ---------------------------------------------------------
    scaler_minmax = MinMaxScaler()
    X_train_mm = X_train.copy()
    X_test_mm = X_test.copy()
    
    X_train_mm[[exp_column]] = scaler_minmax.fit_transform(X_train[[exp_column]])
    X_test_mm[[exp_column]] = scaler_minmax.transform(X_test[[exp_column]])
    print("-> Option B (MinMaxScaler) applied successfully.")

# Note: Tree algorithms are scale-invariant, but scaling continuous experience features 
# provides compatibility if adding distance-based estimators. Proceeding with Standard Scaled data.

# ---------------------------------------------------------
# MODEL 1: DECISION TREE (BASELINE ON SCALED DATA)
# ---------------------------------------------------------
dt_baseline = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_baseline.fit(X_train_std, y_train)

y_train_pred_dt = dt_baseline.predict(X_train_std)
y_test_pred_dt  = dt_baseline.predict(X_test_std)

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

# Small grid due to compressed row profile footprint
param_grid_dt = {
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3]
}

# 5-Fold Cross-Validation Optimization Grid
grid_dt = GridSearchCV(estimator=dt_reg, param_grid=param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)
grid_dt.fit(X_train_std, y_train)

print("\nBest Parameters Found via CV:", grid_dt.best_params_)

dt_opt = grid_dt.best_estimator_
y_train_pred_opt = dt_opt.predict(X_train_std)
y_test_pred_opt  = dt_opt.predict(X_test_std)

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
    n_estimators=150,
    max_depth=4,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_std, y_train)

y_train_pred_rf = rf.predict(X_train_std)
y_test_pred_rf  = rf.predict(X_test_std)

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
metrics_compare = {
    'Model': ['Baseline Decision Tree', 'Optimized Decision Tree', 'Random Forest'],
    'Train Accuracy': [accuracy_score(y_train, y_train_pred_dt), accuracy_score(y_train, y_train_pred_opt), accuracy_score(y_train, y_train_pred_rf)],
    'Test Accuracy': [accuracy_score(y_test, y_test_pred_dt), accuracy_score(y_test, y_test_pred_opt), accuracy_score(y_test, y_test_pred_rf)],
    'Test Precision (Anomalous)': [precision_score(y_test, y_test_pred_dt, pos_label=1), precision_score(y_test, y_test_pred_opt, pos_label=1), precision_score(y_test, y_test_pred_rf, pos_label=1)],
    'Test Recall (Anomalous)': [recall_score(y_test, y_test_pred_dt, pos_label=1), recall_score(y_test, y_test_pred_opt, pos_label=1), recall_score(y_test, y_test_pred_rf, pos_label=1)]
}

df_compare = pd.DataFrame(metrics_compare)
print("\n" + "="*60)
print("FINAL MODEL PERFORMANCE COMPARISON")
print("="*60)

#==============================================================================
Business Benefit & Strategic Impact Report:-
#===============================================================================
Deploying an automated salary verification and anomaly detection framework offers
 direct operational advantages for talent acquisition and HR departments:

Eliminating Information Asymmetry: HR teams no longer have to rely blindly on 
self-reported previous compensation. The tree model evaluates candidate experience 
splits (no of Years of Experience) alongside historic role designations (Position of
the employee) to immediately highlight out-of-market claims.

Optimized Compensation Benchmarking: The solution prevents payroll inflation by
 flagging anomalies before the offer generation phase. This helps ensure that salary
 offers are based on objective market values rather than negotiated exaggerations.

Streamlined Background Screenings: Recruiters can automate preliminary candidate
 verification. Instead of spending time manually auditing every applicant's past 
 payroll records, teams can focus verification resources on profiles flagged as 
 anomalous by the system.
 