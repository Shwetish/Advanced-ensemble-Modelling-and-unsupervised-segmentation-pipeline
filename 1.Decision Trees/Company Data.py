# ==============================================================================
# SALES ANALYSIS FOR A CLOTH MANUFACTURING COMPANY
# USING DECISION TREE & RANDOM FOREST
# ==============================================================================

# ------------------------------------------------------------------------------
# BUSINESS UNDERSTANDING
# ------------------------------------------------------------------------------

# 1. Business Problem Statement:
# A cloth manufacturing company wants to understand
# which product, pricing, and market factors lead to HIGH sales.

# 2. Business Objectives:
# - Classify sales performance (Low -> Better)
# - Identify key drivers influencing sales
# - Support production, pricing, and marketing decisions

# 3. Business Motivation:
# - Understanding demand patterns helps:
# - Avoid over/under production
# - Improve inventory planning
# - Optimize shelf placement and advertising spend

# 4. Constraints:
# - Mixed numeric & categorical data
# - Presence of outliers
# - Highly competitive market (imports)

# 5. Success Criteria:
# Business Success:
# - Actionable insights for sales improvement
# ML Success:
# - Good test accuracy
# - Interpretable decision rules

# -----------------------------------------------------------------------------
# DATA UNDERSTANDING
# ----------------------------------------------------------------------------

'''
Name of Feature          	Description	                    Type
Sales	      Sailing product department	 Target,Quantitative, Continuous
Comp_price	          Competitor Price	            Quantitative, Discrete
Income	              Area Income	                Quantitative, Discrete
Advertising           promotion	                    Quantitative, Discrete
Population	          Company population	        Quantitative, Discrete
Price	              Company Price	                Quantitative, Discrete
ShelveLoc	          Shelf quality per location	Qualitative, Categorical
Age	                  Average Age	                Quantitative, Discrete
Education	          Education Level	            Quantitative, Discrete
Urban	              Urban Area	                Qualitative, Categorical
US	                  United States	                Qualitative, Categorical


Key Business Insights:
- Pricing, shelf placement, and advertising are controllable levers
- Demographic features explain demand variation
- Sales converted to categories for decision-oriented modeling

Key Insights:
- Sales is continuous → converted to categorical
- Mixed data types → encoding required
- Supervised classification problem
'''
#------------------------------------------------------------------------------
# STEP 1: IMPORT REQUIRED LIBRARIES
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set(style="whitegrid")

# ===========================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# Company_Data - Sales Analysis
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("c:/16_Decision_Tree/Company_Data.csv")

df.head()
#--------------------------------------------------------------------------
# 1 First Moment - Mean
#--------------------------------------------------------------------------
df.mean(numeric_only=True)
'''

Inference:
Mean represents the average market condition per store.

• Average Sales -> Typical store performance
• Average Price -> Common pricing strategy
• Average Advertising -> Baseline promotion spend

Used for:
Baseline sales and pricing decisions
'''
#----------------------------------------------------------------------------
#2 Second Moment - Variance & Standard Deviation
#----------------------------------------------------------------------------
df.var(numeric_only=True)
df.std(numeric_only=True)
'''
Inference:
High variance observed in:
- CompPrice
- Income
- Population
- Price and Age

Business Meaning:
Sales performance varies significantly across stores,
Advertising and population size strongly influence demand.
'''
#-----------------------------------------------------------------------------
#3 Third Moment - Skewness
#-----------------------------------------------------------------------------
df.skew(numeric_only=True)
'''
Feature                    Skewness      Business Insight
---    ---    ---
Sales                   Right skewed    Few stores generate very high sales.
Advertising             Right skewed    Few stores spend heavily on promotion.
Price                   Slight skew     Mostly consistent pricing strategies.
Income                  Near symmetric  Balanced customer income distribution.
Population              Mild skew       Mix of small & large market store sizes.

Inference:
Right skew indicates:
High sales and heavy advertising are concentrated in a few stores.
Tree-based models are suitable for such distributions.
'''
#------------------------------------------------------------------------------
# 4 Fourth Moment - Kurtosis
#------------------------------------------------------------------------------
df.kurtosis(numeric_only=True)

'''
Inference:
Sales (-0.08)
Almost symmetrically distributed, indicating balanced sales across the regions.

CompPrice (0.04)
Near symmetric distribution, suggesting consistent competitor pricing.

Income (-1.09)
Left-skewed, meaning more regions fall in the higher income range, with a lower average price.

Advertising (-0.55)
Moderately left-skewed, indicating most stores invest moderately to highly in advertising, with fewer very low spenders.

Population (-1.20)  
Strongly left-skewed, showing more stores operate in high-population areas, with fewer in small markets.  

Price (0.45)  
Moderately right-skewed, indicating a few premium-priced products alongside mostly mid-range pricing.  

Age (-1.13)  
Left-skewed, suggesting customer base is largely middle-aged to older, with fewer very young populations.  

Education (-1.30)  
Strongly left-skewed, indicating higher education levels dominate, with fewer low-education workers.
'''
#----------------------------------------------------------------------------------------------
#Step 8: Histograms (Distribution Analysis)
#----------------------------------------------------------------------------------------------
df.hist(figsize=(14,10), edgecolor='black')
plt.suptitle("Histogram of All Numerical Features")
plt.show()

'''
Sales:
Sales show a slight right skew, indicating most stores have moderate sales, with a few high-
performing outlets.
CompPrice:
Competitor prices are approximately normally distributed, suggesting stable and competitive
market pricing.
Income:
Income is fairly evenly spread across regions, reflecting diverse customer purchasing power.
Advertising:
Advertising expenditure is right-skewed, meaning most stores spend less, while a few 
invest heavily in promotions.
Population appears uniformly distributed ,indicating stores operate acrossboth small and large
markets.
price:
product prices follow anear-normal distibution,suggesting consistent pricing strategy with 
limited extremes
Age:customer age is evenly distributed ,implied demand across multiple age groups.
Education:
Education levels show discrete clusters ,reflecting structurededucation categories
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

Sales:
Few high-value outliers indicate exceptionally high-performing stores, which can heavily 
influence average sales.

CompPrice:
Presence of outliers suggests price variation among competitors, reflecting competitive market
conditions.

Income:
Moderate spread with Limited outliers indicates stable income distribution across regions.

Advertising:
Strong right-side outliers show that only a few stores invest heavily in advertising, while most
spend modestly.

Population:
Very wide range and Large IQR indicate stores operate in both small towns and large cities,
 explaining demand variability.

Price:
Some outliers exist, implying premium and discounted pricing strategies in certain markets.

Age:
Narrow IQR with minimal outliers suggests customer age is relatively consistent across regions.

Education:
Very small spread indicates education levels are fairly uniform, contributing less variability 
compared to other features. 

Business Interpretation:
These represent premium stores, metro markets, or aggressive campaigns Outliers should be
retained.Tree-based models handle them effectively.
'''
#---------------------------------------------------------------------------------------------
# Step 10: Target Variable Distribution (Sales Category)
#--------------------------------------------------------------------------------------------
# Create Sales Category (if not already created)
bins = [0, 5, 10, 15, 20]
labels = ['Low', 'Average', 'Good', 'Better']

df['Sales_cat'] = pd.cut(df['Sales'], bins=bins, labels=labels)

sns.countplot(x='Sales_cat', data=df)
plt.title("Sales Category Distribution")
plt.show()
'''
Inference:
Sales categories are reasonably balanced.
No severe class imbalance.
SMOTE is NOT required.
Why SMOTE is NOT needed (based on the plot)

From the Sales Category Distribution:
Average → majority class

Low and Good → reasonably represented

Better → very small class

This is mild to moderate imbalance, not severe imbalance.

When SMOTE is required

SMOTE is recommended when:

One class is extremely rare (e.g., < 5%)

Model fails to learn minority class

Minority class recall is very poor

Business cost of missing minority

That is not the case here.

Why SMOTE is a bad idea here

Your models are Decision Tree / Random Forest

Tree-based models:

Handle imbalance better than Linear models

Learn from class boundaries, not distance

SMOTE can:

Create synthetic, unrealistic sales patterns 
Increase Overfitting
Dstort Genuine Business distributions
Best practice for this dataset
Instead of SMOTE,do this:

RandomForestClassifier(
    class_weight ='balanced',
    )
'''
#Co-relation Heatmap
# Step 11: Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
'''
Sales vs Price (-0.44)
Moderate negative correlation indicates that higher prices reduce sales, confirming price 
sensitivity.
Sales vs Advertising (0.27)
Positive correlation shows advertising increases sales, though the impact is moderate.
Sales vs Income (0.15)
Weak positive correlation suggests higher-income regions tend to generate slightly higher sales.
Sales vs Age (-0.23)
Negative correlation indicates sales are relatively higher in younger markets.
Sales vs CompPrice (0.06)
very weak corelation shows competitor pricing has limited direct impact on sales
Compprice vs price(0.58)
strong positive corelation indicates pricing strategies closely follow competitors
'''
# Step 12: Scatter Plot (Business Relationship)
sns.scatterplot(x='Price', y='Sales', data=df)
plt.title("Price vs Sales")
plt.show()
'''
Inference:
Higher prices generally lead to lower sales.
Confirms price sensitivity in the market.
'''
# Step 13: PDF & CDF Analysis
for col in df.select_dtypes(include=np.number).columns:
    plt.figure(figsize=(12,4))

# PDF
plt.subplot(1,2,1)
sns.kdeplot(df[col], fill=True)
plt.title(f'PDF of {col}')

# CDF 
plt.subplot(1,2,2)
sorted_vals = np.sort(df[col])
y = np.arange(len(sorted_vals)) / len(sorted_vals)
plt.plot(sorted_vals, y)
plt.title(f'CDF of {col}')
plt.show()
'''
Inference:
PDF shows sales concentration around mid-range values.
CDF helps identify thresholds, e.g.,
80% of stores have sales below a certain level.
Useful for inventory planning.
'''
#Step 14: Pairplot (Feature Interaction)
sns.pairplot(df[['Sales','Price','Advertising','Income','Population']])
plt.show()
'''
Inference:
Advertising shows positive association with sales
Price shows inverse relationship with Sales.
Population amplifies sales potential.

# Step 15: Final EDA Summary
- Dataset is clean and business-realistic
- Sales and Advertising are right-skewed
- Outliers represent high-value business cases
- Price, Advertising, Shelf Location are key drivers

Suitable Models:
Decision Tree
Random Forest
Ensemble Tree Models
'''

#===========================================================================
#Data Preprocessing on Company Data
#===========================================================================
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

sns.set(style="whitegrid")
# It sets a predefined visual theme for all upcoming Seaborn/Matplotlib plots

#----------------------------------------------------------------------------
#step 2: Load Data set
#----------------------------------------------------------------------------

df = pd.read_csv("c:/16_Decision_Tree/Company_Data.csv")

print("Initial Shape:", df.shape)
df.head()

'''
Inference:
Dataset contains retail sales data with
numeric + Categorical features
'''

#--------------------------------------------------------------------------
#Step3: Data Type Check
#--------------------------------------------------------------------------

df.dtypes

'''
Inference:
    Mixed data types present.
    Categorical encoding required.
    '''
#------------------------------------------------------------------------
#step4:Missing value Analysis
#-------------------------------------------------------------------------   
    
print("Missing values:\n", df.isnull().sum())

#---------------------------------------------------------
#Step 5: Duplicate Removal
#---------------------------------------------------------
df.drop_duplicates(inplace=True)
print("After removing duplicates:", df.shape)

'''
Inference:
Ensures no repeated store records.
Prevents biased learning.
'''

#---------------------------------------------------------
#Step 6: Target Variable Creation (Sales Category)
#---------------------------------------------------------
bins = [0, 5, 10, 15, 20]
labels = ['Low', 'Average', 'Good', 'Better']



df['Sales_cat'] = pd.cut(df['Sales'], bins=bins, labels=labels)

# Handle missing values in target
#Not Required

df['Sales_cat'].value_counts()

'''
Inference:
Sales categories are reasonably balanced.
No severe class imbalance.
SMOTE is NOT required.
Why SMOTE is NOT needed (based on the data)

From the Sales Category Distribution:

Average -> majority class

Low and Good -> reasonably represented

Better -> very small class

This is mild to moderate imbalance, not severe imbalance.
When SMOTE is required

SMOTE is recommended when:

One class is extremely rare (e.g., < 5%)

Model fails to learn minority class

Minority class recall is very poor

Business cost of missing minority class is very high
That is not the case here.

Why SMOTE is a bad idea here

Your models are Decision Tree / Random Forest

Tree-based models:

Handle imbalance better than linear models

Learn from class boundaries, not distance

SMOTE can:

Create synthetic, unrealistic sales patterns

Increase overfitting

Distort genuine business distributions
Best practice for this dataset
Instead SMOTE,do this,

RandomForestClassifier(
    class_weight='balanced')
'''
#--------------------------------------------------------------------------------
#step7:- Encoding Categorical Variables
#-------------------------------------------------------------------------------
le = LabelEncoder()

df['ShelveLoc'] = le.fit_transform(df['ShelveLoc'])
df['Urban']     = le.fit_transform(df['Urban'])
df['US']        = le.fit_transform(df['US'])
df['Sales_cat'] = le.fit_transform(df['Sales_cat'])

'''
Why Label Encoding?
Tree-based models do not require one-hot encoding.
Preserves ordinal nature of Shelve Location.
Feature                        Label Encoding           One-Hot Encoding
Output Format                  Single column           Multiplebinary columns
Values Assigned          Integer numbers(0,1,2..)       0 or 1 (binary)
Order Implied?           Yes (creates artificial order)  No order implied
Suitable For             Ordinal categories            Nominal categories
Risk                     Model may assume ranking      No ranking assumption
Dimensionality           No increase                   Increases number of col
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
Outliers present in:
CompPrice
Sales
Price

These represent real high-value business cases.
'''

# ---------------------------------------------------------
# Step 9: Outlier Treatment (Winsorization)
# ---------------------------------------------------------
winsorizer = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['CompPrice', 'Price'])
df[['CompPrice', 'Price']] = winsorizer.fit_transform(df[['CompPrice', 'Price']])

'''
Inference:
Instead of removing data, we capped extreme values.
Keeps the dataset size intact.
Reduces the impact of extreme price fluctuations.
'''

# ---------------------------------------------------------
# Step 10: Skewness Detection
# ---------------------------------------------------------
num_cols = df.select_dtypes(include=np.number).columns
skew_values = df[num_cols].apply(lambda x: skew(x))
skew_values

'''
Inference:
Some features are skewed.
Tree-based models can handle skewness naturally.
'''
# ---------------------------------------------------------
# IMPORTANT NOTE: Log Transformation
# ---------------------------------------------------------
'''
Do we need Log Transformation?

Case 1: Tree-Based Models (DT, RF, Bagging)
NO

Why?
* Trees split based on order, not distribution
* Skewness does not affect tree performance
* Winsorization is sufficient (even optional)

Case 2: Linear / Distance-Based Models
YES (Log + Scaling required)
'''
# ---------------------------------------------------------
# Step 11: Feature Scaling
# ---------------------------------------------------------
'''
Feature scaling is NOT required for:
Decision Tree
Random Forest
Bagging
Boosting

Reason:
Tree models are scale-invariant.
'''

'''
Your target Sales_cat has a class (most likely Better) with only 2 record
stratify=y requires at least 2 samples per class
Stratified splitting tries to keep class proportions -> impossible with 1 sampl
If you will check
y.value_counts()
'''

# Verify distribution
print(df['Sales_cat'].value_counts())
df = df[df['Sales_cat'] != 1]
#here 1=better
print(df['Sales_cat'].value_counts())

X = df.drop(columns=['Sales', 'Sales_cat'])
y = df['Sales_cat']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    #stratify=y
)
# ---------------------------------------------------------
# MODEL 1: DECISION TREE (BASELINE)
# ---------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize Decision Tree model
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
# Fixed: added closing quote and closing parenthesis
print(classification_report(y_test, y_test_pred))


#----------------------------------------------------------------------------------------------
#DECISION TREE OPTIMIZATION (OVERTITTING CONTROL)
#----------------------------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#STEP 1: Define a Regularized Decision Tree
dt_reg = DecisionTreeClassifier(
criterion='entropy',
random_state=42
)

#STEP 2: Hyperparameter Grid (Controls Complexity)
param_grid = {
'max_depth': [3, 5, 7, 9],
'min_samples_split': [10, 20, 30],
'min_samples_leaf': [5, 10, 15]
}

#Why these parameters?

#max_depth -> Limits tree growth

#----------------------------------------------------------------------------------------------
#STEP 3: GridSearchCV (Cross-Validated Optimization)
#-------------------------------------------------------------------------------------------------
grid_dt = GridSearchCV(
estimator=dt_reg,
param_grid=param_grid,
cv=5,
scoring='accuracy',
n_jobs=-1
)

grid_dt.fit(X_train, y_train)

#STEP 4: Best Model from Grid Search
print("Best Parameters:", grid_dt.best_params_)

#STEP 5: Train Optimized Decision Tree
dt_opt = grid_dt.best_estimator_

#Predictions
y_train_pred_opt = dt_opt.predict(X_train)
y_test_pred_opt = dt_opt.predict(X_test)

#Performance
print("Optimized DT Train Accuracy :", accuracy_score(y_train, y_train_pred_opt))
print("Optimized DT Test Accuracy :", accuracy_score(y_test, y_test_pred_opt))

#Confusion Matrix
print("\nConfusion Matrix (Test Data):\n")
print(confusion_matrix(y_test, y_test_pred_opt))

#Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_test_pred_opt))

'''
The Decision Tree has improved (earlier it was near 100% train accuracy)
But Decision Trees still have high variance
The dataset is small + noisy + multi-class
DT alone is not powerful enough to generalize well
This is expected behavior, not a failure.

WHY THIS IS HAPPENING (Very Important)
1 Dataset Charecteristics

2 Decision Trees are HIGH-VARIANCE MODELS
Even with:
max_depth
min_samples_leaf
min_samples_split
A single tree:
Still memorizes local patterns
Still unstable to small data changes
This is a known limitation, not bad tuning.

3 Accuracy is a harsh metric here     
3-class problem

Random guessing ≈ 33%

55% test accuracy is meaningfully better than chance

But yes – not production-grade yet

WHAT IS THE CORRECT NEXT STEP?

You do NOT fight overfitting further with a single Decision Tree.
You change the model class.

'''
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
n_estimators=300,
max_depth=6,
min_samples_leaf=10,
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
This is exactly the right moment to pause and
interpret, not panic.
What you're seeing now is NOT a tuning failure
— it's a data-limited ceiling.
'''


