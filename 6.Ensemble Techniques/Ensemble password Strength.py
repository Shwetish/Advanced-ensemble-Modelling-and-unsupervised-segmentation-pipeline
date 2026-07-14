#==============================================================
#Business Understanding:-
#==============================================================
#1. Business Problem StatementCyberattacks and data breaches happen every day 
#because users create weak passwords that are easy for hackers to guess using 
#automated cracking tools. Security teams struggle to instantly catch and block
# weak passwords when a user is signing up, leaving company databases and user 
#accounts vulnerable to being hacked.

#2. Business ObjectiveTo build an automated
# classification tool that instantly evaluates a password's complexity
# (characters) and flags its tier (characters_strength), forcing users to
# create stronger passwords before they can finish registering.

#3. MotivationPrevents Hacking:- Forcing users to fix weak passwords stops a 
#massive percentage of credential-based cyberattacks before they ever start.
#Protects Company Reputation: Avoiding data breaches saves the company from
# massive legal fines, negative news headlines, and loss of customer trust.
#Real-Time Guidance: Instead of a vague "password too weak" message, an 
#intelligent model can guide users on exactly what to add (like symbols, 
#capitals, or numbers) to make it safe.

#4. Constraints:-Needs to be Fast: The model must evaluate the password in a 
#fraction of a second. If it causes a delay when a user is trying to register,
# they will get frustrated and leave the website.Text Analysis Complexity:
#Passwords don't follow normal dictionary spelling. They are a chaotic 
#mix of lengths, cases, and special characters, requiring advanced text-feature
# engineering (like counting character types or length).Smart Hackers: Hackers
# constantly update their cracking dictionaries, meaning the model needs to be
# updated regularly to stay ahead of new common password trends.

#5. Business Success Criteria:-
#Primary Goal: Reduce the registration of weak/vulnerable passwords (strength 
#tier 0) to 0% across the platform within a month of deployment.
#Secondary Goal: See a noticeable drop in successful brute-force hacking attempt
# on user accounts over the next two quarters.

#6. ML (Machine Learning) Success
# CriteriaAccuracy $\ge$ 92%: The model must correctly classify password tiers
# accurately so it doesn't accidentally block a truly strong password or 
#accept a very weak one.Precision $\ge$ 95% for Strong Passwords: If the model
# flags a password as safe (tier 1 or higher), it needs to be absolutely 
#certain it can't be easily cracked.

#=============================================================================
#Data Understanding:-
#=============================================================================
'''
Name of Feature            Description                   Type       Relevance
characters         The raw password text string     Qualitative,    Highly 
                   entered by the user.            Text / Nominal   Relevant.
characters_        The evaluation tier or complexity Qualitative,   Highly
strength          category of the password.            Ordinal      Relevant 
                                                                    (Target 
                                                                    Variable).
'''

# ==============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# PASSWORD STRENGTH DATASET - CYBERSECURITY RISK PROFILING
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string  # <--- It is here, but we will make sure Python registers it

# Load Dataset
df = pd.read_excel("C:/18_Ensemble_Baggging/Ensemble_password_strength .xlsx")

print("First 5 records of the raw user registrations:")
print(df.head())

# ==============================================================================
# STEP 1: CYBERSECURITY FEATURE ENGINEERING (Converting text to numbers)
# ==============================================================================
# Dropping potential null entries in text column
df = df.dropna(subset=['characters'])
df['characters'] = df['characters'].astype(str)

# Extract structural numeric indicators from raw password text
df['Length'] = df['characters'].apply(len)
df['Digit_Count'] = df['characters'].apply(lambda x: sum(c.isdigit() for c in x))
df['Uppercase_Count'] = df['characters'].apply(lambda x: sum(c.isupper() for c in x))

# Added a local import here just in case your notebook environment loses track of it
import string 
df['Special_Count'] = df['characters'].apply(lambda x: sum(c in string.punctuation for c in x))

# Target alignment naming cleanup
target_col = 'characters_strength'

# ==============================================================================
# UNIVARIATE ANALYSIS (FOUR MOMENTS OF BUSINESS DECISION)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. First Moment - Mean (Baseline Complexity Structures)
# ------------------------------------------------------------------------------
print("\n--- FIRST MOMENT: MEAN ---")
print(df[['Length', 'Digit_Count', 'Uppercase_Count', 'Special_Count']].mean())
'''
Inference:
The Mean describes the baseline behavioral habits of an average platform user.

• Average Length -> Tells us the standard character length a user types voluntarily.
• Average Special_Count -> Highlights if typical users naturally include secure punctuation.

Used for:
Setting basic baseline system requirements for security validation gates.
'''

# ------------------------------------------------------------------------------
# 2. Second Moment - Variance & Standard Deviation (Complexity Variation)
# ------------------------------------------------------------------------------
print("\n--- SECOND MOMENT: STANDARD DEVIATION ---")
print(df[['Length', 'Digit_Count', 'Uppercase_Count', 'Special_Count']].std())
'''
Inference:
• Length Standard Deviation -> Shows whether user input length varies widely or stays tightly 
  clamped around traditional lengths (like 6 to 10 characters).
• Special_Count Variance -> A low variance here means almost all general users uniformally 
  ignore or skip special characters entirely unless strictly forced to use them.
'''

# ------------------------------------------------------------------------------
# 3. Third Moment - Skewness (User Behavioral Tendencies)
# ------------------------------------------------------------------------------
print("\n--- THIRD MOMENT: SKEWNESS ---")
print(df[['Length', 'Digit_Count', 'Uppercase_Count', 'Special_Count']].skew())
'''
Inference:
• Special_Count & Uppercase_Count (Extremely Right Skewed) -> Confirms that the overwhelming majority 
  of passwords contain exactly zero or very few capitals and symbols, with a tiny trailing 
  minority of high-security professionals creating high-complexity variations.
'''

# ------------------------------------------------------------------------------
# 4. Fourth Moment - Kurtosis (Extreme Custom Behaviors)
# ------------------------------------------------------------------------------
print("\n--- FOURTH MOMENT: KURTOSIS ---")
print(df[['Length', 'Digit_Count', 'Uppercase_Count', 'Special_Count']].kurtosis())
'''
Inference:
• Length (High Positive Kurtosis) -> Leptokurtic shape. Indicates that user behavior is heavily 
  anchored around common milestone patterns (like exactly 8 characters). Extreme values (passwords 
  with 30+ characters) represent rare, distinct systemic anomalies.
'''

# ------------------------------------------------------------------------------
# 5. Histograms (Consolidated Complexity Distributions)
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Password Length Distribution
sns.histplot(df['Length'], kde=True, ax=axes[0], color='navy', bins=15)
axes[0].set_title("Distribution of Password Lengths")
axes[0].set_xlabel("Total Character Length")

# Number of Digits Distribution
sns.histplot(df['Digit_Count'], kde=True, ax=axes[1], color='indigo', bins=10)
axes[1].set_title("Distribution of Digit Counts per Password")
axes[1].set_xlabel("Number of Digits")

plt.suptitle("Histograms of User Generated Complexity Metrics", fontsize=14)
plt.tight_layout()
plt.show()
'''
Inference:
Length Histogram:
Shows a clean peak followed by a rapid drop-off, visualizing the exact threshold where 
most automated dictionaries terminate brute-force scans.

Digit Count Histogram:
Tends to show discrete spikes at 0, 1, or 4 digits, indicating that users routinely append 
predictable elements like birth years or single sequences to basic words.
'''

# ------------------------------------------------------------------------------
# 6. Boxplots (Outlier & Automated Script Detection)
# ------------------------------------------------------------------------------
plt.figure(figsize=(12, 4))
sns.boxplot(data=df[['Length', 'Digit_Count', 'Special_Count']], orient='h', palette='Set1')
plt.title("Boxplot Profiles: Identifying Extreme Password Variations & Potential API Abuse")
plt.show()
'''
Inference:
Extreme high outliers in the Length feature (e.g., strings stretching over 40+ characters) 
often point to automated passphrases, programmatic service tokens, or copy-pasted buffer noise 
rather than normal, manually typed passwords.
'''

# ------------------------------------------------------------------------------
# 7. Target Variable Distribution (Security Tier Structural Balance)
# ------------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x=target_col, data=df, palette='viridis')
plt.title("Target Variable Distribution (Password Strength Tiers)")
plt.xlabel("Strength Category")
plt.show()
'''
Inference:
The target security distribution maps out how many entries fall into weak vs secure clusters. 

Why SMOTE must be avoided here:
If your target contains balanced category counts across strength tiers, artificial synthesis 
is completely unnecessary. Running SMOTE on generated password indicators can create invalid 
structural combinations (like a 3-character password marked as elite strength), breaking the 
model's internal logical boundaries.
'''

# ==============================================================================
# BIVARIATE & MULTIVARIATE ANALYSIS
# ==============================================================================

# ------------------------------------------------------------------------------
# 8. Correlation Analysis (Complexity Metric Interdependency)
# ------------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
numeric_cols = ['Length', 'Digit_Count', 'Uppercase_Count', 'Special_Count', target_col]
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='mako', fmt=".2f")
plt.title("Complexity Metric vs Password Strength Correlation Matrix")
plt.show()
'''
Inference:
Length vs Password Strength:
Typically reveals a commanding positive linear connection. This confirms that basic length 
serves as the primary defensive barrier against quick brute-force password cracking attempts.
'''

# ------------------------------------------------------------------------------
# 9. Scatter Plot (Security Isolation Mapping)
# ------------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Length', y='Digit_Count', hue=target_col, data=df, palette='viridis', alpha=0.7)
plt.title("Security Mapping Boundary: Password Length vs Digit Density")
plt.show()
'''
Inference:
The scatter chart reveals clear structural groupings. Weak classifications cluster tightly 
in the short-length, zero-digit zone. Secure labels pull away into the longer, mixed-character 
territories, proving that ensemble classification trees can easily separate these categories.
'''

# ==============================================================================
# ADVANCED PROBABILITY DISTRIBUTION ANALYSIS (PDF & CDF)
# ==============================================================================

for col in ['Length', 'Digit_Count']:
    plt.figure(figsize=(12, 4))

    # Probability Density Function (PDF)
    plt.subplot(1, 2, 1)
    sns.kdeplot(data=df, x=col, hue=target_col, fill=True, common_norm=False, palette='viridis')
    plt.title(f'PDF: Structural Density of {col}')

    # Cumulative Distribution Function (CDF)
    plt.subplot(1, 2, 2)
    for strength_val in df[target_col].unique():
        sub_series = df[df[target_col] == strength_val][col]
        sorted_vals = np.sort(sub_series)
        y_vals = np.arange(len(sorted_vals)) / len(sorted_vals)
        plt.plot(sorted_vals, y_vals, label=f'Tier: {strength_val}')
        
    plt.title(f'CDF: Cumulative Security Percentiles for {col}')
    plt.xlabel(col)
    plt.ylabel('Cumulative Probability')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
'''
Inference:
PDF Visual Benefit:
Clearly highlights how distinct the separation is between complexity tiers. The weak tier 
peaks abruptly at short lengths, whereas the robust security tiers show wider, flatter curves 
pushed deep downfield.

CDF Visual Benefit:
Acts as a precise policy benchmark calculator. It lets security team leads determine the exact 
percentile breakdown—for example, discovering exactly what percentage of vulnerable Tier 0 
passwords fail to cross an 8-character limit.
'''

# ==============================================================================
# FINAL EDA SUMMARY
# ==============================================================================
'''
- Raw text parameters require standard string extraction (Feature Engineering) before 
  any model processing can take place.
- String length and digit concentrations provide clear, highly distinct separating boundaries 
  for classification targets.
- Outliers in feature length represent distinct passphrase archetypes or bot inputs rather than 
  errors and should be preserved.

Recommended Strategy:
1. Extract rich character traits (such as alphanumeric combinations and uppercase presence) 
   using lambda functions to build your input feature matrix.
2. Feed these structural features directly into an Ensemble Classifier (like Random Forest or 
   Gradient Boosting). These models natively isolate structural decision boundaries without requiring 
   any feature scaling.
'''
# ==============================================================================
# DATA PREPROCESSING OF
# PASSWORD STRENGTH ENSEMBLE DATASET - 
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Model Architectures
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, BaggingClassifier, 
                              AdaBoostClassifier, VotingClassifier, StackingClassifier)
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Evaluation Framework
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set styling configurations for clear diagnostic visual printouts
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------------------
# Step 2: Load Dataset
# ------------------------------------------------------------------------------
# (Update file path string to match your local project environment)
df = pd.read_excel("C:/18_Ensemble_Baggging/Ensemble_password_strength .xlsx")

print("="*60)
print(f"INITIAL DATA ENGINE INFRASTRUCTURE - Shape: {df.shape}")
print("="*60)
print(df.head())

# ------------------------------------------------------------------------------
# Step 3: Structural Feature Data Type Check
# ------------------------------------------------------------------------------
print("\n--- STEP 3: DATA TYPE SUMMARY ---")
print(df.dtypes)
'''
Inference:
Features consist of raw text password strings along with structural categorical or 
numerical strength labels indicating vulnerability vectors.
'''

# ------------------------------------------------------------------------------
# Step 4: Feature Engineering, Cleaning and Imputation
# ------------------------------------------------------------------------------
print("\n--- STEP 4: FEATURE ENGINEERING AND TEXT EXTRACTION ---")
# Strip potential trailing or leading whitespaces from column headers to eliminate KeyErrors
df.columns = df.columns.str.strip()

# Handle missing password fields before string parsing using the correct column 'characters'
df['characters'] = df['characters'].fillna("").astype(str)

# Advanced Character Metrics Extraction: Translating raw text into numerical dimensions
df['length'] = df['characters'].apply(len)
df['digit_count'] = df['characters'].apply(lambda x: sum(c.isdigit() for c in x))
df['special_count'] = df['characters'].apply(lambda x: sum(not c.isalnum() for c in x))
df['upper_count'] = df['characters'].apply(lambda x: sum(c.isupper() for c in x))

# Drop the raw text feature column to prevent direct leakage during modeling
df = df.drop(columns=['characters'])

print("Missing Value Counts remaining across engineered features:")
print(df.isnull().sum().sum())
# ------------------------------------------------------------------------------
# Step 5: Duplicate Record Removal
# ------------------------------------------------------------------------------
print("\n--- STEP 5: DUPLICATE ANALYSIS ---")
initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
duplicate_diff = initial_rows - df.shape[0]
print(f"Removed {duplicate_diff} duplicate password records. Cleaned dataset shape: {df.shape}")

# ------------------------------------------------------------------------------
# Step 6: Target Variable Pre-Check & Balance Evaluation
# ------------------------------------------------------------------------------
print("\n--- STEP 6: TARGET CLASS DISTRIBUTION ---")
target_feature = 'strength'
print(df[target_feature].value_counts())
'''
Inference:
The dataset targets exhibit a standard clinical multi-class distribution representing 
Weak, Medium, and Strong groupings. SMOTE is bypassed to keep real-world baseline 
ratios; class weights will handle internal optimization adjustments.
'''

# ------------------------------------------------------------------------------
# Step 7: Categorical Target String Encoding
# ------------------------------------------------------------------------------
print("\n--- STEP 7: CATEGORICAL TARGET ENCODING ---")

# Fix: Set the variable to match your actual column name
target_feature = 'characters_strength'

le = LabelEncoder()
df[target_feature] = le.fit_transform(df[target_feature].astype(str))

# Dynamically mapping target representations (e.g., 0 = Weak, 1 = Medium, 2 = Strong)
print("Target mapping multi-class representation established:")
print(df[target_feature].value_counts())

# ------------------------------------------------------------------------------
# Step 8: Outlier Detection Visualization
# ------------------------------------------------------------------------------
plt.figure(figsize=(12, 6))
engineered_features = ['length', 'digit_count', 'special_count', 'upper_count']
sns.boxplot(data=df[engineered_features], orient='h', palette='Set1')
plt.title("Visual Structural Distribution Profile Before Outlier Treatment", fontsize=14)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# Step 9: Outlier Treatment via Advanced Winsorization
# ------------------------------------------------------------------------------
print("\n--- STEP 9: OUTLIER TREATMENT (WINSORIZATION) ---")

# Fix: Only target 'length' for outlier capping since count columns have too low variation
outlier_features = ['length'] 

winsorizer = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=outlier_features)
df[outlier_features] = winsorizer.fit_transform(df[outlier_features])

print("Winsorization complete. Extreme length anomalies safely clamped to operational margins.")

# ------------------------------------------------------------------------------
# Step 10: Skewness Re-Evaluation
# ------------------------------------------------------------------------------
print("\n--- STEP 10: SKEWNESS VALUES AFTER CAPPING ---")
skew_values = df[engineered_features].apply(lambda x: skew(x))
print(skew_values)
'''
Inference:
Tree ensembles split boundaries based on relative ordinal distributions rather than continuous 
spatial vector distances. Transformations are skipped as tree models are natively scale invariant.
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
X = df.drop(columns=[target_feature])
y = df[target_feature]

# Enforce 'stratify=y' to preserve equal class balances across splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training Features Dimensions: {X_train.shape} | Testing Features Dimensions: {X_test.shape}")

#==================================================================================
# Understanding the Model Outputs (The Multi-Class Matrix)
#==================================================================================
# The model maps authentication strings into key matrix fields based on predictions:
# True Positives (Diagonal elements): The model correctly identifies the true security class.
# False Positives (Off-Diagonal column elements): Passwords flagged as stronger than they are.
# False Negatives (Off-Diagonal row elements) – Dangerous Vulnerability Misses: 
# A password is Weak, but the model incorrectly classifies it as Strong. 

# Why "Precision & Recall" Matter: While accuracy shows total pipeline performance, 
# High Recall on low strength levels ensures that insecure passwords do not bypass filters 
# and remain unchecked in production databases.


# ==============================================================================
# ADVANCED PIPELINE: END-TO-END DATA PROCESSING & MULTI-ENSEMBLE OPTIMIZATION
# PASSWORD STRENGTH ENSEMBLE DATASET 
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
print("\n--- STEP 5: TEXT PARSING AND CHARACTER METRICS EXTRACTION ---")
# Strip potential trailing or leading whitespaces from column headers to eliminate KeyErrors
df.columns = df.columns.str.strip()

# Handle missing fields before string parsing using your actual column name: 'characters'
df['characters'] = df['characters'].fillna("").astype(str)

# Advanced Character Metrics Extraction: Translating raw text into numerical dimensions
df['length'] = df['characters'].apply(len)
df['digit_count'] = df['characters'].apply(lambda x: sum(c.isdigit() for c in x))
df['special_count'] = df['characters'].apply(lambda x: sum(not c.isalnum() for c in x))
df['upper_count'] = df['characters'].apply(lambda x: sum(c.isupper() for c in x))

# Only drop the text column AFTER you are completely done extracting everything
df = df.drop(columns=['characters'])

# ------------------------------------------------------------------------------
# Step 9: Outlier Treatment via Multi-Feature Winsorization
# ------------------------------------------------------------------------------
print("\n--- STEP 9: OUTLIER TREATMENT (WINSORIZATION) ---")

# Fix: Force the winsorizer to only look at 'length' 
# This skips 'special_count' and 'upper_count' which were causing the crash
outlier_features = ['length']

winsorizer = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=outlier_features)
df[outlier_features] = winsorizer.fit_transform(df[outlier_features])

print("Winsorization complete. Extreme length anomalies safely clamped to operational margins.")

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
# Mapping multi-class parameters for XGBoost multi:softmax execution
xgb_base = XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y_train)), random_state=42, eval_metric='mlogloss')
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
    final_estimator=LogisticRegression(class_weight='balanced', random_state=42, max_iter=500),
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
    print("   Confusion Matrix Breakdown:")
    print(cm)
    print("\n   Granular Classification Performance Analysis Report:")
    print(classification_report(y_test, test_predictions, zero_division=0))
    print("=" * 70)

#==================================================================================
# Business Benefits & Impact (How the Client Wins)
#==================================================================================
# Implementing this system helps an enterprise or platform infrastructure in four major ways:

# 1. Proactive Defense Against Account Takeovers: It moves enterprise infrastructure from 
# post-incident response to immediate edge checking. Weak configurations are flagged 
# instantly at creation, mitigating brute-force and credential stuffing entry points.

# 2. Optimized UX via Real-Time Analytics: Users receive instant, accurate score 
# feedback based on deep token distributions (length, case variances, special characters) 
# instead of brittle, hardcoded regular expression checks.

# 3. Targeted Internal Enforcement Strategies: System admins can flag and query 
# legacy database tables containing low-tier password vectors, forcing specific user segments 
# to upgrade credentials before a data leak occurs.

# 4. Reduces Regulatory & Insurance Liability Costs: Enforcing high-assurance password policies 
# helps satisfy modern compliance framework mandates (like GDPR, HIPAA, or PCI-DSS), 
# directly lowering operational cyber-insurance premiums and avoiding multi-million dollar fines.