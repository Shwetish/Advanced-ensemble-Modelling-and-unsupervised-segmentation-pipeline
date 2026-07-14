#================================================================
#Business Understanding:-
#==================================================================
#1. Business Problem Statement:-
#Medical teams cannot manually keep up with screening every 
#patient for diabetes risks before they get sick. Right now, 
#healthcare providers are reactive—treating diabetes after a 
#patient shows severe symptoms. This delay causes preventable 
#long-term complications for patients and results in massive,
# unnecessary treatment costs for hospitals.

#2. Business Objective:-
#To build an automated early-warning screening tool that analyzes
# routine health vitals (like glucose levels, age, and blood 
#pressure) to accurately flag high-risk patients before full 
#diabetes develops.

#3. Motivation:-
#Saves Lives: Catching diabetes early allows patients to reverse
# or manage the condition through simple diet and lifestyle
# changes.
#Reduces Costs: Preventive care is significantly cheaper for both
# patients and healthcare networks than treating chronic, advanced
# kidney or heart complications later.
#Flawless Screening: It removes human oversight, ensuring no high-
#risk patient slips through the cracks during busy clinic hours.

#4. Constraints:-
#Data Quality (Zeros): Vital tracking features often contain 
#missing values or accidental placeholders (like a recorded blood
# pressure or insulin level of 0), which can break mathematical 
#predictions if not handled carefully.
#High Cost of False Negatives (The Biggest Risk): Telling a sick
# patient they are "Healthy" (False Negative) is dangerous 
#because they will leave without treatment. The model must 
#prioritize minimizing this specific error.
#Privacy Regulations: The data handles personal, private patient
# health vitals, meaning strict data-handling security must be 
#maintained throughout the code framework.

#5. Business Success Criteria:-
#Primary Goal: Successfully screen and route at least 85% of 
#high-risk diabetic patients into early preventative care 
#programs within the first quarter of deployment.
#Secondary Goal: Noticeably reduce the average cost per patient
# by minimizing emergency-room visits related to undiagnosed 
#diabetic spikes.

#6. ML (Machine Learning) Success Criteria:-
#Recall Score ≥ 85% to 90% (Crucial Metric): High recall ensures
# the model successfully flags the vast majority of true diabetes
# cases, keeping dangerous False Negatives to an absolute minimum.
#Overall Accuracy ≥ 80%: The model should maintain high overall 
#correctness so doctors can rely on its flags without being 
#overwhelmed by false alarms.

#=============================================================================
#Data Understanding:-
#=============================================================================
'''
Name of Feature     	Description	          Type	     Relevance
Number of times  Total number of pregnancies Quantitative, Highly
 pregnant	        the patient has had.	 Discrete	 Relevant.
Plasma glucose   Blood sugar level measured
 concentration	 after a 2-hour oral glucose Quantitative,Highly
                  tolerance test.	         Discrete	 Relevant.
Diastolic blood  Blood pressure reading in 
                 mm Hg when the heart rests  Quantitative, 	Highly
pressure	      between beats.	         Discrete   Relevant.
Triceps skin     Fat layer thickness
fold thickness	 measured on the back of     Quantitative,  Highly
                 the arm in mm.              Discrete	  Relevant.
2-Hour serum   Amount of insulin hormone 
insulin	       in the blood after 2 hours   Quantitative,  Highly
               in mu U/ml.	                Discrete      Relevant.
Body mass   Body weight score calculated    Quantitative, Highly
index	   relative to the patient's height. Continuous	  Relevant.
Diabetes      A calculated genetic score
pedigree       based on the patient's       Quantitative,  Highly
function	  family medical history.	     Continuous	  Relevant.
Age (years)	   The current age of           Quantitative, Highly
                the patient.	            Discrete	  Relevant.
Class        Indicates whether the patient  Qualitative,   Highly
variable	 has diabetes or not (YES or NO). Nominal	  Relevant
                                                          (Target 
                                                         Variable)
'''
#--------------------------------------------------------------------------------------
#Part1:-Python Implementation
#--------------------------------------------------------------------------------------
#Exploratory Data Analysis (EDA)
#--------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data 
df = pd.read_csv("C:/18_Ensemble_Baggging/Diabeted_Ensemble .csv")
df.columns = df.columns.str.strip()
#----------------------------------------------------------------------------------
# Summary
#-------------------------------------------------------------------------------
print(df.describe())
'''
#Inference: This summary provides a quick mathematical snapshot of the metrics (such
 as average age or maximum glucose). It identifies impossible numbers immediately, 
such as a minimum value of 0 for Blood Pressure or BMI.
'''
#-------------------------------------------------------------------------------
# Univariate Analysis
#--------------------------------------------------------------------------------
df.hist(figsize=(12, 10), color='teal', edgecolor='black')
plt.suptitle("Univariate Analysis: Distribution of Vitals")
plt.tight_layout()
plt.show()
'''
Inference: Looking at each feature individually, parameters like Age, Pregnancies, 
and Insulin are right-skewed. Most patients are younger, with fewer pregnancies and
 lower insulin values, tapering down to a small group of high-value cases.
'''
#-----------------------------------------------------------------------------------
# Bivariate Analysis
#-----------------------------------------------------------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='YlGnBu', fmt=".2f")
plt.title("Bivariate Analysis: Feature Correlation Matrix")
plt.show()
'''
Strongest Connection: Plasma glucose concentration has the highest positive 
correlation with the outcome variable. This means as blood sugar rises, the likelihood
of having diabetes increases significantly.

Other Links: Clear connections exist between Body Mass Index (BMI) and skin fold 
thickness, as well as between Age and the number of times pregnant.

Value: Glucose is the most critical feature for the machine learning model to track
when identifying high-risk patients.
'''
#------------------------------------------------------------------------------------
# Checking interaction between the top marker (Glucose) and target
#------------------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x='Class variable', y='Plasma glucose concentration', data=df, palette='Set2')
plt.title("Glucose Levels vs. Diabetes Class Variable")
plt.show()
'''
Inference: The heatmap proves that Plasma glucose concentration has the strongest 
direct relationship with whether a patient has diabetes. The boxplot confirms that 
diabetic patients ('YES') maintain significantly higher glucose levels than healthy 
ones.
'''
#------------------------------------------------------------------------------------
#Data Pre-processing
#-------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer

# Data Cleaning (Turning invalid clinical zeros into NaN)
zero_cols = ['Plasma glucose concentration', 'Diastolic blood pressure', 
             'Triceps skin fold thickness', '2-Hour serum insulin', 'Body mass index']
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)

# Fill gaps using Median to prevent outlier distortion
imputer = SimpleImputer(strategy='median')
df[zero_cols] = imputer.fit_transform(df[zero_cols])

# Encode Class variable (NO -> 0, YES -> 1)
le = LabelEncoder()
df['Class variable'] = le.fit_transform(df['Class variable'].astype(str))

# Outlier Treatment (Winsorization)
outlier_features = ['Number of times pregnant', 'Diastolic blood pressure', '2-Hour serum insulin', 
                    'Body mass index', 'Diabetes pedigree function', 'Age (years)']
winsorizer = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=outlier_features)
df[outlier_features] = winsorizer.fit_transform(df[outlier_features])

print("Cleaned Data Shape:", df.shape)
'''
Inference: Impossible zero-values are replaced with the median value so they do not 
corrupt calculations. Extreme statistical outliers are clamped safely using an IQR 
winsorizer so the learning algorithms are not misdirected by anomalies.
'''
#-------------------------------------------------------------------------------------
#Model Building & Hyperparameter Tuning
#-------------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Split features and target
X = df.drop(columns=['Class variable'])
y = df['Class variable']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#-------------------------------------------------------------------------------------
# Data Scaling
#-------------------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#-------------------------------------------------------------------------------------
# Bagging, Boosting, Voting, and Stacking via GridSearchCV
#-------------------------------------------------------------------------------------
# --- Bagging (Random Forest Framework) ---
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_param = {'n_estimators': [50, 100], 'max_depth': [4, 6, 8]}
rf_grid = GridSearchCV(rf, rf_param, cv=5, scoring='recall', n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)
best_rf = rf_grid.best_estimator_

# --- Boosting (AdaBoost & XGBoost) ---
ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=42)
ada_param = {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1, 1.0]}
ada_grid = GridSearchCV(ada, ada_param, cv=5, scoring='recall', n_jobs=-1)
ada_grid.fit(X_train_scaled, y_train)
best_ada = ada_grid.best_estimator_

xgb = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
xgb.fit(X_train_scaled, y_train)

# --- Voting Ensemble ---
voting_clf = VotingClassifier(estimators=[('rf', best_rf), ('ada', best_ada), ('xgb', xgb)], voting='soft')
voting_clf.fit(X_train_scaled, y_train)

# --- Stacking Ensemble ---
stacking_clf = StackingClassifier(
    estimators=[('rf', best_rf), ('ada', best_ada), ('xgb', xgb)],
    final_estimator=LogisticRegression(), cv=5
)
stacking_clf.fit(X_train_scaled, y_train)

# Performance Evaluation and Comparison
models = {
    "Bagging (Random Forest)": best_rf,
    "AdaBoost (Boosting)": best_ada,
    "XGBoost (Boosting)": xgb,
    "Voting Classifier": voting_clf,
    "Stacking Classifier": stacking_clf
}

print("\n" + "="*60 + "\nMODEL ACCURACIES & CONFUSION MATRICES\n" + "="*60)
for name, model in models.items():
    preds = model.predict(X_test_scaled)
    print(f"\n▶ {name}")
    print(f"Accuracy Score: {accuracy_score(y_test, preds):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    

'''
Inference: The grid search automatically selects the best architectural parameters 
focused directly on maximizing the Recall score. By scaling the inputs, algorithms 
reliably compare metrics with differing scales, such as Age and Insulin. The Voting
and Stacking models combine the structural opinions of all baseline trees to smooth 
out individual model blind spots, decrease false negatives, and produce a stable 
prediction tool.
'''
#------------------------------------------------------------------------------
#R Implementation
#Exploratory Data Analysis (EDA) & Data Pre-processing
#------------------------------------------------------------------------------

library(tidyverse)
library(caret)
library(randomForest)
library(adabag)
library(xgboost)

# Load data
df <- read.csv("C:/18_Ensemble_Baggging/Diabeted_Ensemble .csv")
colnames(df) <- trimws(colnames(df))

# Summary
print(summary(df))

# Pre-processing: Fix physiological zero values
invalid_vars <- c("Plasma.glucose.concentration", "Diastolic.blood.pressure", 
                  "Triceps.skin.fold.thickness", "2.Hour.serum.insulin", "Body.mass.index")

for(col in invalid_vars) {
  df[[col]][df[[col]] == 0] <- NA
}

# Apply median values to fill NA gaps
for(col in invalid_vars) {
  median_val <- median(df[[col]], na.rm = TRUE)
  df[[col]][is.na(df[[col]])] <- median_val
}

# Cap statistical outliers safely via IQR logic
cap_values <- function(x) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = TRUE)
  H <- 1.5 * IQR(x, na.rm = TRUE)
  x[x < (qnt[1] - H)] <- qnt[1] - H
  x[x > (qnt[2] + H)] <- qnt[2] + H
  return(x)
}
numeric_idx <- sapply(df, is.numeric)
df[numeric_idx] <- lapply(df[numeric_idx], cap_values)

# Turn label into standard factor variable
df$Class.variable <- as.factor(df$Class.variable)
'''
Inference: The R code isolates missing values masked as 0, sets them to the median 
value of the variable, and trims extreme data spikes on the outer boundaries.
'''
#------------------------------------------------------------------------------------
#Model Building (Ensembles)
#-------------------------------------------------------------------------------------
set.seed(42)
splitIndex <- createDataPartition(df$Class.variable, p = .8, list = FALSE)
train_set <- df[splitIndex,]
test_set  <- df[-splitIndex,]

# Cross Validation control parameter + auto-scaling configuration
train_ctrl <- trainControl(method="cv", number=5, preProc = c("center", "scale"))

# --- Bagging ---
grid_rf <- expand.grid(.mtry = c(2, 4, 6))
model_bagging <- train(Class.variable ~ ., data = train_set, method = "rf", 
                       trControl = train_ctrl, tuneGrid = grid_rf)

# --- Boosting ---
model_boosting <- train(Class.variable ~ ., data = train_set, method = "AdaBoost.M1", 
                        trControl = train_ctrl, tuneLength = 2)

# Evaluation 
bag_preds <- predict(model_bagging, test_set)
bst_preds <- predict(model_boosting, test_set)

print("=== BAGGING EVALUATION ===")
print(confusionMatrix(bag_preds, test_set$Class.variable)$table)

print("=== BOOSTING EVALUATION ===")
print(confusionMatrix(bst_preds, test_set$Class.variable)$table)
'''
Inference: The trainControl function automatically handles the data scaling process. 
The matrix summaries display exactly how many true positive cases are successfully 
captured versus cases where the model errors.
'''
#-------------------------------------------------------------------------------------
#Business Impact:-
#-------------------------------------------------------------------------------------
Business Impact and Client Benefits
Early Clinical Identification: Changes clinical protocol from reactive treatment to 
proactive early monitoring. High-risk patients are accurately flagged before 
complications emerge.

Operational Expense Reduction: Mitigates massive, preventable expenditures linked to 
emergency critical care or long-term disease complications, optimizing hospital 
network resource allocation.

Automated Safety Net: Minimizes the risk of human oversight during busy clinical 
intake windows by automatically screening multiple diagnostic markers simultaneously.
