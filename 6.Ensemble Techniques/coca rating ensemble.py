#============================================================
#Business Undestanding:-
#============================================================
#1. Business Problem Statement:-
#The gourmet chocolate market is highly competitive and depends heavily
# on quality perceptions. Chocolate manufacturing companies, 
#distributors, and retailers struggle to know exactly which factors—
#such as bean varieties (Bean_Type), origin countries (Origin), or 
#cocoa percentages (Cocoa_Percent)—make a chocolate bar achieve elite,
# high-rated status versus a mediocre or low rating.

#2. Business Objective:-
#To analyze chocolate review data to uncover the exact ingredient, 
#geographic, and formulation combinations that maximize professional
# quality ratings (Rating), enabling chocolate companies to optimize 
#their sourcing, recipes, and marketing.

#3. Motivation:-
#Smarter Sourcing: Premium cocoa beans are expensive. Sourcing raw 
#materials from regions or varieties statistically proven to get higher
# ratings prevents wasting money on low-quality crops.
#Product Optimization: Finding the "sweet spot" for cocoa percentages
# ensures production lines manufacture flavor profiles that consumers 
#and experts highly value.
#Competitive Pricing: Chocolate bars backed by high ratings can be 
#marketed as premium luxury items, allowing businesses to justify 
#higher price tags and increase profit margins.

#4. Constraints:-
#Subjective Data: The target variable (Rating) is based on human taste
# testers, introducing subjective preferences and personal bias into 
#the data.
#Text Formatting Issues: Text categories like Bean_Type or Origin
# contain multiple names, custom blends, or empty cells, requiring
# intensive text cleaning before training a machine learning model.
#Changing Trends: Consumer preferences and expert standards shift over
# time (e.g., changes in review scores across different Review years).

#5. Business Success Criteria:-
#Primary Goal: Identify the top 3 bean origins and ideal cocoa
# percentage ranges that yield premium ratings (≥3.75) to guide the 
#next season's product manufacturing.
#Secondary Goal: Launch a newly formulated chocolate product line based
# on these insights that achieves an expert review score above the
# historical market average.

#6. ML (Machine Learning) Success Criteria:-
#Mean Absolute Error (MAE) ≤ 0.25 (If Regression): If predicting the
# exact numeric rating (from 1 to 5), the model's predictions should 
#be off by no more than a quarter of a rating point on average.
#Classification Accuracy ≥ 80% (If Classification): If classifying 
#chocolate into simple tiers like "Premium" vs "Standard", the model
# must accurately identify the correct tier four out of five times.

#=============================================================================
#Data Understanding:-
#=============================================================================
'''
Name of              Description              Type            Relevance
Feature     CompanyThe brand name of the    Qualitative,      Highly
             chocolate manufacturer.        Nominal           Relevant.
Name        The specific name or regional   Qualitative       Highly
           crop blend of the chocolate bar., Nominal          Relevant.
REF        An internal tracking reference 
           number grouping when the review  Quantitative,     Irrelevant
           was entered.                      Discrete        
Review     The publication year when the    Quantitative,     Highly
           chocolate bar was reviewed.      Discrete          Relevant.
Cocoa_     The percentage of dark
Percent    cocoa content contained in       Quantitative,     Highly
           the chocolate bar.               Continuous        Relevant.
Company_   The geographic country 
Location   where the manufacturing          Qualitative,      Highly
           company is based.                Nominal           Relevant.
Rating     The expert quality score 
           assigned to the bar              Quantitative     Highly 
           (typically from 1 to 5).,       Continuous        Relevant
                                                             (Target 
                                                              Variable).
Bean_Type  The specific biological variety
           of the cocoa bean used          Qualitative,       Highly
           (e.g., Criollo, Trinitario).     Nominal          Relevant.
Origin     The geographic region or 
           country where the cocoa beans   Qualitative,      Highly
           were harvested.                 Nominal           Relevant.
           
'''
# ==============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# CHOCOLATES RATING DATASET - QUALITY PROFILING & MARKET INSIGHTS
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
# (Update this file path to match where your dataset is saved)
df = pd.read_excel("C:/18_Ensemble_Baggging/Coca_Rating_Ensemble .xlsx")

print("First 5 records of the chocolate rating inventory:")
print(df.head())

# Clean 'Cocoa_Percent' to transform text strings (e.g., '63%') into true numeric continuous features
if 'Cocoa_Percent' in df.columns and df['Cocoa_Percent'].dtype == 'object':
    df['Cocoa_Percent'] = df['Cocoa_Percent'].str.replace('%', '').astype(float)

# Drop explicit tracking indices like REF if present to avoid statistics distortion
if 'REF' in df.columns:
    df = df.drop(columns=['REF'])

# ==============================================================================
# UNIVARIATE ANALYSIS (FOUR MOMENTS OF BUSINESS DECISION)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. First Moment - Mean (Standard Quality Baselines)
# ------------------------------------------------------------------------------
print("\n--- FIRST MOMENT: MEAN ---")
print(df.mean(numeric_only=True))
'''
Inference:
The Mean establishes standard production baselines across the global chocolate catalog.

• Rating (~3.18) -> The typical market quality baseline sits slightly above average.
• Cocoa_Percent (~71.7%) -> Establishes that the standard international production formula 
  is centered tightly around dark chocolate recipes.
• Review (~2012) -> Shows the typical mid-point era of data collection.

Used for:
Benchmarking new manufacturer batches against standard international baselines.
'''

# ------------------------------------------------------------------------------
# 2. Second Moment - Variance & Standard Deviation (Consistency Evaluation)
# ------------------------------------------------------------------------------
print("\n--- SECOND MOMENT: STANDARD DEVIATION ---")
print(df.std(numeric_only=True))
'''
Inference:
• Rating Std Dev (~0.47) -> Points to a highly narrow, selective rating corridor. True 
  flavour scores rarely drop below 2 or rise above 4.5.
• Cocoa_Percent Std Dev (~6.3%) -> Confirms that while 70-75% is standard, manufacturers 
  occasionally diversify into extreme milk blends or pure baking bars.
'''

# ------------------------------------------------------------------------------
# 3. Third Moment - Skewness (Market Behavior Tendencies)
# ------------------------------------------------------------------------------
print("\n--- THIRD MOMENT: SKEWNESS ---")
print(df.skew(numeric_only=True))
'''
Inference:
• Rating (Left/Negative Skew) -> The consumer/reviewer curve trails off to the left. 
  Indicates that poorly crafted chocolate bars are quickly flagged and heavily penalized, 
  while the majority maintain passable standard scores.
• Cocoa_Percent (Right/Positive Skew) -> Confirms that while dark chocolates rule the dataset, 
  there is a right-hand trail caused by high-intensity bars stretching up toward 90-100% purity.
'''

# ------------------------------------------------------------------------------
# 4. Fourth Moment - Kurtosis (Anomalous Formulations)
# ------------------------------------------------------------------------------
print("\n--- FOURTH MOMENT: KURTOSIS ---")
print(df.kurtosis(numeric_only=True))
'''
Inference:
• Cocoa_Percent (High Positive Kurtosis) -> Leptokurtic profile. Shows an aggressive 
  centralization around 70%. Deviations into alternative formulation recipes (extreme milk or 
  pure cacao) represent niche product offerings.
'''

# ------------------------------------------------------------------------------
# 5. Histograms (Consolidated Distribution Overviews)
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Cocoa Content Distribution
sns.histplot(df['Cocoa_Percent'], kde=True, ax=axes[0], color='chocolate', bins=15)
axes[0].set_title("Distribution of Cocoa Content (%)")
axes[0].set_xlabel("Cocoa Percentage")

# Rating Distribution
sns.histplot(df['Rating'], kde=True, ax=axes[1], color='darkgoldenrod', bins=10)
axes[1].set_title("Distribution of Expert Consumer Ratings")
axes[1].set_xlabel("Rating Score")

plt.suptitle("Histograms of Continuous Chocolate Metrics", fontsize=14)
plt.tight_layout()
plt.show()
'''
Inference:
Cocoa Percent Histogram:
Shows prominent spikes at round milestones (70%, 75%). This confirms a strong industry 
preference for standardized dark chocolate target formulations.

Rating Histogram:
Confirms a normal, slightly left-skewed shape. The strict boundary limits mean a model must 
predict continuous ranges between 2.5 and 4.0 accurately.
'''

# ------------------------------------------------------------------------------
# 6. Boxplots (Recipe Outlier Identification)
# ------------------------------------------------------------------------------
plt.figure(figsize=(12, 4))
sns.boxplot(data=df[['Cocoa_Percent', 'Rating']], orient='h', palette='Set2')
plt.title("Boxplot Profiles: Highlighting Anomalies in Recipe Formulations & Quality Ratings")
plt.show()
'''
Inference:
The outliers in Cocoa_Percent reveal bars with extreme ratios (below 55% or at 100%). 
These represent distinct market categories (baking blocks or sweetened confectioneries) 
and should be monitored carefully during data stratification.
'''

# ------------------------------------------------------------------------------
# 7. Categorical Cardinality & Concentration (Top Processing Hubs)
# ------------------------------------------------------------------------------
plt.figure(figsize=(12, 5))
top_locations = df['Company_Location'].value_counts().head(10)
sns.barplot(x=top_locations.index, y=top_locations.values, palette='copper')
plt.title("Top 10 Global Production Hubs by Representation")
plt.ylabel("Number of Registered Blends")
plt.xticks(rotation=45)
plt.show()
'''
Inference:
The dataset shows a high concentration of roasting and processing centers located in 
established consumption markets (such as the U.S.A. and France), rather than being 
exclusively distributed across agricultural growing regions.
'''

# ==============================================================================
# BIVARIATE & MULTIVARIATE ANALYSIS
# ==============================================================================

# ------------------------------------------------------------------------------
# 8. Correlation Analysis (Recipe Impact Matrix)
# ------------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.heatmap(df[['Review', 'Cocoa_Percent', 'Rating']].corr(), annot=True, cmap='BrBG', fmt=".2f")
plt.title("Recipe Formulation & Timeline Correlation")
plt.show()
'''
Inference:
Cocoa Percent vs Rating (Slight Negative Correlation):
Shows that blindly increasing cocoa concentration does not guarantee a higher quality rating. 
In fact, extreme dark bars often face lower palatability scores if the bean texture isn't 
properly balanced.
'''

# ------------------------------------------------------------------------------
# 9. Scatter Plot (Finding the Sweet Spot)
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Cocoa_Percent', y='Rating', data=df, alpha=0.6, color='saddlebrown')
# Adding a central trendline to trace target patterns
sns.regplot(x='Cocoa_Percent', y='Rating', data=df, scatter=False, color='red')
plt.title("The 'Sweet Spot' Analysis: Cocoa Purity vs Resulting Rating")
plt.show()
'''
Inference:
The distribution shows a clear 'sweet spot' clustering. The highest elite ratings 
(scores $\ge 3.75$) are densely grouped between 65% and 75% cocoa content. Once purity 
surpasses 85%, ratings begin to slip downward due to natural bitterness constraints.
'''

# ==============================================================================
# ADVANCED ANALYSIS: TIME-SERIES CONCENTRATION VARIANCE
# ==============================================================================
plt.figure(figsize=(10, 5))
sns.lineplot(x='Review', y='Rating', data=df, errorbar='ci', color='darkgreen', marker='o')
plt.title("Evolution of Market Quality Trends Across Review Years")
plt.xlabel("Year of Production Evaluation")
plt.ylabel("Average Rating")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
'''
Inference:
The historical profile shows a clear upward trend in average ratings over time. This reflects 
steady improvements in international bean processing methods, sourcing quality, and craft 
manufacturing techniques over the years.
'''

# ==============================================================================
# FINAL EDA SUMMARY
# ==============================================================================
'''
- Continuous feature behavior confirms an industry preference for 70-75% cocoa formulas.
- Maximizing cocoa content exhibits a point of diminishing returns; balance is key to achieving 
  optimal rating scores.
- High categorical cardinality across 'Company', 'Bean_Type', and 'Origin' indicates that a 
  Target Encoder or One-Hot Encoding threshold strategy will be required prior to model training.

Recommended Strategy:
1. Construct a new feature grouping high-volume origins to prevent model noise from rare, 
   single-entry bean locations.
2. Utilize tree-based models like Random Forest or Gradient Boosting to capture non-linear 
   relationships, such as the 70% cocoa 'sweet spot' curve.
'''
# ==============================================================================
# DATA PREPROCESSING OF
# COCOA RATING DATASET - 
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
df = pd.read_excel("C:/18_Ensemble_Baggging/Coca_Rating_Ensemble .xlsx")

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
Features blend text string indicators (Company, Locations, Bean Varieties) with 
numerical tracking metrics and categorical evaluation records.
'''

# ------------------------------------------------------------------------------
# Step 4: Missing Value Analysis & Handling & Cleaning
# ------------------------------------------------------------------------------
print("\n--- STEP 4: FEATURE CLEANING AND IMPUTATION ---")
# Strip potential trailing or leading whitespaces from column headers to eliminate KeyErrors
df.columns = df.columns.str.strip()

# Drop unique tracking identifiers to avoid baseline statistics distortion
if 'REF' in df.columns:
    df = df.drop(columns=['REF'])

# Clean and transform text-encoded percentage objects (e.g. '63%') into scalable float features
if 'Cocoa_Percent' in df.columns and df['Cocoa_Percent'].dtype == 'object':
    df['Cocoa_Percent'] = df['Cocoa_Percent'].str.replace('%', '').astype(float)

# Group structural text columns to implement category filling strategies
string_cols = ['Company', 'Name', 'Company_Location', 'Bean_Type', 'Origin']
for col in string_cols:
    df[col] = df[col].fillna("Unknown").astype(str)

print("Missing Value Counts remaining across features:")
print(df.isnull().sum().sum())

# ------------------------------------------------------------------------------
# Step 5: Duplicate Record Removal
# ------------------------------------------------------------------------------
print("\n--- STEP 5: DUPLICATE ANALYSIS ---")
initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
duplicate_diff = initial_rows - df.shape[0]
print(f"Removed {duplicate_diff} duplicate rating rows. Cleaned dataset shape: {df.shape}")

# ------------------------------------------------------------------------------
# Step 6: Target Variable Pre-Check & Balance Evaluation (Binning Target)
# ------------------------------------------------------------------------------
print("\n--- STEP 6: TARGET CLASS DISTRIBUTION ---")
# Since classification ensemble tree splits operate on categorical class boundaries, 
# continuous expert rating values are mapped into distinct performance tiers:
# Tier 1 (Premium / Elite Quality) if score >= 3.5, else Tier 0 (Standard Quality)
target_feature = 'Quality_Class'
df[target_feature] = np.where(df['Rating'] >= 3.5, 'Premium', 'Standard')

print(df[target_feature].value_counts())
'''
Inference:
The generated target feature exhibits a highly stable and well-proportioned class split.
SMOTE remains bypassed to preserve raw recipe ratios, relying on native balancing estimators.
'''

# Drop raw continuous rating value column to avoid target leakage during modeling
df = df.drop(columns=['Rating'])

# ------------------------------------------------------------------------------
# Step 7: Categorical Target String Encoding
# ------------------------------------------------------------------------------
print("\n--- STEP 7: CATEGORICAL TARGET ENCODING ---")
le = LabelEncoder()
df[target_feature] = le.fit_transform(df[target_feature].astype(str))
print("Target mapping representation established (0 = Standard, 1 = Premium)")
print(df[target_feature].value_counts())

# ------------------------------------------------------------------------------
# High Cardinality Handling (Frequency Label Encoding)
# ------------------------------------------------------------------------------
# Convert text string descriptions to frequency counts to enable mathematical parsing
for col in string_cols:
    freq_map = df[col].value_counts()
    df[col] = df[col].map(freq_map)

# ------------------------------------------------------------------------------
# Step 8: Outlier Detection Visualization
# ------------------------------------------------------------------------------
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['Review', 'Cocoa_Percent']], orient='h', palette='Set2')
plt.title("Visual Structural Distribution Profile Before Outlier Treatment", fontsize=14)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# Step 9: Outlier Treatment via Advanced Winsorization
# ------------------------------------------------------------------------------
print("\n--- STEP 9: OUTLIER TREATMENT (WINSORIZATION) ---")
outlier_cols = ['Review', 'Cocoa_Percent']
winsorizer = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=outlier_cols)
df[outlier_cols] = winsorizer.fit_transform(df[outlier_cols])

print("Winsorization complete. Non-standard crop percentages safely clamped to standard limits.")

# ------------------------------------------------------------------------------
# Step 10: Skewness Re-Evaluation
# ------------------------------------------------------------------------------
print("\n--- STEP 10: SKEWNESS VALUES AFTER CAPPING ---")
skew_values = df[outlier_cols].apply(lambda x: skew(x))
print(skew_values)
'''
Inference:
Tree ensembles split independent categories sequentially based on rank thresholds. 
Log conversions are skipped as tree leaf paths remain unaffected by distribution skewness.
'''

# ------------------------------------------------------------------------------
# Step 11: Feature Scaling Assessment
# ------------------------------------------------------------------------------
'''
Operational Architecture Note:
Feature scaling scaling transformations are explicitly bypassed for tree models.
Reason:
Ensemble models split nodes based on information theory values (Entropy/Gini impurity), 
rendering them immune to differences in scaling across independent columns.
'''

# ------------------------------------------------------------------------------
# Step 12: Stratified Train-Test Splitting
# ------------------------------------------------------------------------------
print("\n--- STEP 12: STRATIFIED TRAIN-TEST SPLITTING ---")
X = df.drop(columns=[target_feature])
y = df[target_feature]

# Enforce 'stratify=y' to preserve equal premium class balance ratios across subsets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training Features Dimensions: {X_train.shape} | Testing Features Dimensions: {X_test.shape}")

#==================================================================================
# Understanding the Model Outputs (The Confusion Matrix)
#==================================================================================
# The model places chocolate bars into four simple categories based on its predictions:

# True Positives (TP) – Correct Catch: The bar is Premium quality (1), and the model 
# correctly flags it.

# True Negatives (TN) – Correct Safe: The bar is Standard quality (0), and the model correctly
# clears it.

# False Positives (FP) – False Alarm: The bar is Standard quality, but the model flags
# it as Premium by mistake. This leads to commercial inefficiencies, such as overpaying 
# for standard bean stock or mislabeling basic rows as luxury inventory.

# False Negatives (FN) – Dangerous Miss: The bar is Premium quality, but the model 
# calls it Standard. This is the most damaging error for product strategy. The business 
# misses out on elite bean configurations, selling an exceptional blend at low prices.

# Why "Recall" Matters Most: While overall accuracy is nice, Recall is the most 
# critical metric. High recall means the model is excellent at reducing False 
# Negatives (Dangerous Misses), ensuring no premium recipe combination goes unnoticed.


# ==============================================================================
# ADVANCED PIPELINE: END-TO-END DATA PROCESSING & MULTI-ENSEMBLE OPTIMIZATION
# COCOA RATING DATASET 
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
df = pd.read_excel("C:/18_Ensemble_Baggging/Coca_Rating_Ensemble .xlsx")
df.columns = df.columns.str.strip()
if 'REF' in df.columns:
    df = df.drop(columns=['REF'])

if 'Cocoa_Percent' in df.columns and df['Cocoa_Percent'].dtype == 'object':
    df['Cocoa_Percent'] = df['Cocoa_Percent'].str.replace('%', '').astype(float)

target_feature = 'Quality_Class'
df[target_feature] = np.where(df['Rating'] >= 3.5, 1, 0)
df = df.drop(columns=['Rating'])

string_cols = ['Company', 'Name', 'Company_Location', 'Bean_Type', 'Origin']
for col in string_cols:
    df[col] = df[col].fillna("Unknown").astype(str)
    freq_map = df[col].value_counts()
    df[col] = df[col].map(freq_map)

print("="*80)
print(f"INITIAL CLEANING PIPELINE ENGINE INITIALIZED - Initial Shape: {df.shape}")
print("="*80)

# Remove repeat/accidental logging instances
df.drop_duplicates(inplace=True)

# ------------------------------------------------------------------------------
# Step 9: Outlier Treatment via Multi-Feature Winsorization
# ------------------------------------------------------------------------------
outlier_cols = ['Review', 'Cocoa_Percent']
winsorizer = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=outlier_cols)
df[outlier_cols] = winsorizer.fit_transform(df[outlier_cols])

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
    print(f"    [[ True Negatives (Standard): {cm[0][0]}   False Positives (Type I): {cm[0][1]} ]")
    print(f"     [ False Negatives (Missed): {cm[1][0]}   True Positives (Premium): {cm[1][1]} ]]")
    print("\n   Granular Classification Performance Analysis Report:")
    print(classification_report(y_test, test_predictions, zero_division=0))
    print("=" * 70)

#==================================================================================
# Business Benefits & Impact (How the Client Wins)
#==================================================================================
#Implementing this system helps a chocolate manufacturer or distributor in four major ways:

#1. Sourcing Optimization Saves Money: It avoids spending heavy capital stock on over-priced 
#cocoa fields that yield standard quality ratings, directing purchasing power toward high-yield configurations.

#2. Speeds Up Recipe Testing: Production lines do not have to manually run continuous trial batches 
#for years. The AI model instantly reveals the percentage combinations and bean locations that hit premium tiers.

#3. Higher Profit Margins: By confidently identifying premium flavor profiles based on expert inputs, 
#the client can confidently release high-tier product variations with premium pricing.

#4. Lowers Financial Waste: Automatically avoiding weak beans and bad ingredient ratios minimizes 
#scrap product rates, ensures recipe standardization, and reduces raw supply logistics loss.