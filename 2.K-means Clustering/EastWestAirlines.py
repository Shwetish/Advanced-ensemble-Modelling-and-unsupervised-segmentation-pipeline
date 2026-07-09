#----------------------------------------------------------------------------
#Business Understanding
#----------------------------------------------------------------------------

#1.Business Problem Statement
#"The airline cannot efficiently maximize customer loyalty because passengers are
#treated as a single group, leading to generic marketing spend, underutilized reward
#points, and a high risk of losing premium travelers to competitors."

#2. Business Objective
#To discover hidden, distinct traveler groups within the frequent flyer database using
#unsupervised learning (clustering) so the marketing team can launch precision
#retention campaigns.

#3. Business Constraints
#Budget Limits: Reward points, free lounge access vouchers, and flight upgrades cost
#money; they cannot be given to everyone.
#Data Limits: The dataset lacks explicit demographic details (like age or occupation)
#or ticket profit margins, meaning groupings must rely purely on flight behaviors and
#mile balances.

#4. Business Success Criteria
#Successfully segment the airline's customer base into distinct, actionable groups
#(e.g., "Elite Business Frequent Flyers", "Casual Holiday Travelers", "Inactive
#Account Accumulators").
#Achieve a minimum 10% increase in campaign conversion rates and passenger point
#redemptions within 6 months of launching targeted cluster offers.

#5. Machine Learning Success Criteria
#Build an unsupervised clustering model (such as K-Means or Hierarchical Clustering)
#that yields a high Silhouette Score (greater than 0.5) or a clear, distinct elbow
#point on a WCSS graph.
#Ensure the generated clusters show statistically distinct feature distributions
#(e.g., clear, distinct boundaries in Balance, Bonus_miles, and Flight_trans_12) with
# zero overlapping profiles.

#-----------------------------------------------------------------------------------------------------------------------
# Data Understanding (Corrected & Aligned Schema)
#-----------------------------------------------------------------------------------------------------------------------
"""
Column Name          Description                                              Type                    Relevance
-----------------------------------------------------------------------------------------------------------------------
ID#                  Unique identification no assigned to each customer       Qualitative, Nominal    Low
                     (Row Tracker).                                           (Identifier)

Balance              The total number of points a customer currently has      Quantitative, Discrete  High
                     saved up in their account.

Qual_miles           Miles earned by actually flying that help you level      Quantitative, Discrete  High
                     up to VIP status (Silver/Gold/etc).

cc1_miles            Miles earned using the primary co-branded credit         Quantitative, Ordinal   Medium
                     card (Tiers 1-5).

cc2_miles            Miles earned using the secondary co-branded credit       Quantitative, Ordinal   Medium
                     card (Tiers 1-5).

cc3_miles            Miles earned using the third co-branded credit           Quantitative, Ordinal   Medium
                     card (Tiers 1-5).

Bonus_miles          Miles earned from special offers, retail partners,       Quantitative, Discrete  High
                     or sign-up deals instead of flying.

Bonus_trans          How many times the customer used deals or partner        Quantitative, Discrete  High
                     offers to get extra points.

Flight_miles_12      Number of actual miles physically flown by the           Quantitative, Discrete  High
                     customer over the past 12 months.

Flight_trans_12      Number of individual flight trips booked and             Quantitative, Discrete  High
                     completed over the past 12 months.

Days_since_enroll    How long (in days) the customer has been a member        Quantitative, Discrete  High
                     of the airline's loyalty club.

Award?               Has the customer ever used their points for a free       Quantitative, Binary    High
                     flight or reward? (1=Yes, 0=No).
"""



#============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
#============================================================================

#============================================================================
# 1. Basic Exploration
#============================================================================
print("--- First 5 Records ---")
print(df.head())

print("\n--- Last 5 Records ---")
print(df.tail())

print("\n--- Dataset Dimensional Shape ---")
print(df.shape)

print("\n--- Structural Information ---")
print(df.info())

print("\n--- Basic Descriptive Summary ---")
print(df.describe())
"""
Inference:
- Dataset contains customer loyalty behavior information.
- Space consists purely of numerical features; no raw categorical text columns are present.
- Features are appropriate and structurally ready for clustering.
"""

#============================================================================
# 2. Missing Value Analysis
#============================================================================
print("\n--- Missing Value Counts ---")
print(df.isnull().sum())
"""
Inference:
- Clean dataset containing zero missing values across all monitored columns.
- No data imputation transformations are needed.
"""

#============================================================================
# 3. Duplicate Value Check
#============================================================================
print("\n--- Total Duplicate Rows ---")
print(df.duplicated().sum())
"""
Inference:
- Zero duplicate records detected, confirming that each row is a unique customer.
- No row deletions are required.
"""

#============================================================================
# 4. Data Types
#============================================================================
print("\n--- Data Column Types ---")
print(df.dtypes)
"""
Inference:
- Most columns are saved as clean integer/numeric data types.
- Highly optimized configuration for distance-based machine learning algorithms.
"""

#============================================================================
# 5. First Four Moments of Distribution
#============================================================================

# 5.1 First Moment -> Mean
print("\n--- First Moment: Mean ---")
print(df.mean(numeric_only=True))
"""
Inference:
- Establishes expected baseline averages for traveler metrics.
- High average balances signal heavy point accumulation behavior.
"""

# 5.2 Second Moment -> Variance & Standard Deviation
print("\n--- Second Moment: Variance ---")
print(df.var(numeric_only=True))

print("\n--- Second Moment: Standard Deviation ---")
print(df.std(numeric_only=True))
"""
Inference:
- High standard deviations indicate that consumer behavior varies significantly.
- Features like Balance and Bonus_miles display massive spread, requiring future scaling.
"""

# 5.3 Third Moment -> Skewness
print("\n--- Third Moment: Skewness ---")
print(df.skew(numeric_only=True))
"""
Inference:
- Strong positive skewness values confirm a long right tail across variables.
- Most frequent flyers have low balances, while a tiny pocket of elite flyers pulls the distribution right.
"""

# 5.4 Fourth Moment -> Kurtosis
print("\n--- Fourth Moment: Kurtosis ---")
print(df.kurt(numeric_only=True))
"""
Inference:
- Sharp, positive leptokurtic metrics confirm heavy tails.
- This highlights the clear presence of high-value outlier VIP shoppers.
"""

#============================================================================
# 6. Univariate Analysis
#============================================================================
# Histogram Grid Configuration
df.hist(figsize=(15, 12), color='steelblue', edgecolor='black')
plt.suptitle("Univariate Histograms of Airline Metrics", fontsize=16)
plt.tight_layout()
plt.show()
"""
Inference:
- Visual distributions confirm non-normality and high density spikes near zero.
"""

# Global Boxplot Configuration
plt.figure(figsize=(15, 10))
sns.boxplot(data=df.drop(columns=['ID#']))  # Dropping ID# for visual scalability
plt.xticks(rotation=90)
plt.title("Visual Outlier Distributions Across Features")
plt.tight_layout()
plt.show()
"""
Inference:
- Highlights extreme outlier points belonging to high-spending airline customers.
"""

#============================================================================
# 7. Bivariate Analysis
#============================================================================
# Correlation Matrix Plotting
corr = df.corr(numeric_only=True)

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Interaction Correlation Matrix")
plt.tight_layout()
plt.show()
"""
Inference:
- Strong linear correlations uncover structural redundancies:
  * Balance vs. Bonus_miles
  * Flight_miles_12 vs. Flight_trans_12
"""

# Pairplot Matrix
sns.pairplot(df[['Balance', 'Bonus_miles', 'Flight_miles_12', 'Days_since_enroll']])
plt.suptitle("Bivariate Interactions for Core Cluster Drivers", y=1.02)
plt.show()

#============================================================================
# 8. PDF (Probability Density Function)
#============================================================================
plt.figure(figsize=(8, 4))
sns.kdeplot(df['Balance'], fill=True, color='purple')
plt.title("PDF of Balance")
plt.xlabel("Balance")
plt.show()
"""
Inference:
- Displays continuous probability density peaks where consumer balances are concentrated.
"""

#============================================================================
# 9. CDF (Cumulative Distribution Function)
#============================================================================
x = np.sort(df['Balance'])
y = np.arange(len(x)) / float(len(x))

plt.figure(figsize=(8, 4))
plt.plot(x, y, color='darkorange', lw=2)
plt.xlabel("Balance")
plt.ylabel("CDF")
plt.title("CDF of Balance")
plt.show()
"""
Inference:
- Quantifies exact customer threshold splits. For example, it tracks what percentage of accounts sit under a selected milestone score.
"""

#============================================================================
# 10. Outlier Detection (Interquartile Range - IQR Metric)
#============================================================================
print("\n--- Outlier Counts Per Column via IQR ---")
for col in df.columns[1:]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")
"""
Inference:
- Loyalty databases naturally hold highly stretched tails from top-tier fliers.
- Deleting these would clear out high-value data patterns; soft capping should be utilized instead.
"""

#============================================================================
# DATA PRE-PROCESSING
#============================================================================

#============================================================================
# 11. Drop Irrelevant Feature
#============================================================================
df1 = df.drop(['ID#'], axis=1)
print("\n--- Matrix After ID Feature Removal ---")
print(df1.head())
"""
Inference:
- ID is only a customer identifier.
- It has no business meaning for clustering.
- Removing it improves clustering quality.
"""

#============================================================================
# 12. Outlier Treatment Using Winsorization
#============================================================================
for col in df1.columns:
    df1[col] = winsorize(df1[col], limits=[0.05, 0.05])
"""
Inference:
- Winsorization reduces extreme values without deleting records.
- Important because airline VIP customers are valuable observations.
"""

#============================================================================
# 13. Skewness Check Post-Winsorization
#============================================================================
print("\n--- Post-Winsorization Skewness Matrix ---")
print(df1.skew())
"""
Inference:
- Positive skewness indicates many low-frequency travelers and few extremely loyal customers.
- Common behavior in airline datasets.
"""

#============================================================================
# 14. Feature Scaling
#============================================================================
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df1)

scaled_df = pd.DataFrame(scaled_data, columns=df1.columns)
print("\n--- Standardized Scaled DataFrame (Head) ---")
print(scaled_df.head())
"""
Inference:
- All variables converted to same scale.
- Improves K-Means cluster performance.
"""

#============================================================================
# 15. Check Mean and Standard Deviation After Scaling
#============================================================================
print("\n--- Standardized Feature Calculated Means ---")
print(scaled_df.mean().round(4))

print("\n--- Standardized Feature Calculated Deviations ---")
print(scaled_df.std().round(4))
"""
Inference:
- Mean approximately 0.
- Standard deviation approximately 1.
- Data properly standardized.
"""

#============================================================================
# MODEL BUILDING & CLUSTER OPTIMIZATION
#============================================================================

#============================================================================
# 16. Scree Plot / Elbow Method to Identify Optimum Cluster Cap
#============================================================================
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_df)
    wcss.append(kmeans.inertia_)

# Generating visual Scree Plot
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--', color='indigo')
plt.title('Scree Plot (Elbow Curve) for Optimal K Selection')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.show()
"""
Inference:
- The curve drops sharply from K=1 to K=4, and significantly flattens out afterwards.
- The 'elbow joint' points directly to k=4 or k=5 as the mathematical optimum.
"""

#============================================================================
# 17. Cluster Validation & Profile Comparisons (K=4 vs K=5 Approaches)
#============================================================================
# Appending K=4 Model Output Labels
kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster_K4'] = kmeans_4.fit_predict(scaled_df)

# Appending K=5 Model Output Labels
kmeans_5 = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster_K5'] = kmeans_5.fit_predict(scaled_df)

print("\n=== Cluster Behavioral Profiles for K=4 Approach ===")
summary_k4 = df.drop(columns=['ID#', 'Cluster_K5']).groupby('Cluster_K4').mean()
print(summary_k4.round(2))

print("\n=== Cluster Behavioral Profiles for K=5 Approach ===")
summary_k5 = df.drop(columns=['ID#', 'Cluster_K4']).groupby('Cluster_K5').mean()
print(summary_k5.round(2))
"""
Inference & Validation Summary:
- Approach K=4 creates highly interpretable customer segments:
  * Cluster 0 represents premium flyers with frequent bookings.
  * Cluster 1 tracks credit-card spenders who collect points via transactions.
  * Cluster 2 identifies historical enrollees who have turned dormant.
  * Cluster 3 maps low-balance, newly acquired members at risk of churning.
- Splitting into K=5 breaks down the low-balance segment into redundant groups without
  offering distinct, actionable new customer personas. K=4 is preferred for corporate strategy.
"""



'''
#============================================================================
# 18. BENEFITS AND IMPACT OF THE SOLUTION FOR EAST WEST AIRLINES
#============================================================================
"""
By shifting away from uniform marketing approaches and adopting the K=4 customer 
segmentation model, East West Airlines can achieve several clear business benefits:

1. Stop Wasting Marketing Money (Targeted Offers)
   - Old Way: The airline sends the same generic discount email to all 4,000+ 
              customers. This wastes marketing capital and triggers high unsubscribe rates.
   - New Way: The marketing team offers premium credit card upgrades exclusively to 
              Cluster 1 (Card Spenders) and routes targeted flight flash sales exclusively 
              to Cluster 0 (Frequent Flyers).
   - Result: Maximizes sales conversion rates while dropping marketing costs significantly.

2. Bring Back "Sleepy" Customers (Wake-Up Campaigns)
   - The Insight: The model isolates Cluster 2—historical enrollees who accumulated 
                  miles years ago but have completely stopped booking flights.
   - The Action: Launch an automated "We Miss You" campaign offering a localized, 
                 short-term mile redemption discount if they book within 30 days.
   - Result: Recovers easy revenue from dormant accounts and eliminates outstanding 
             mileage liabilities from the balance sheet.

3. Prevent Customers from Leaving (Churn Prevention)
   - The Insight: Cluster 3 isolates new or casual holiday flyers holding low balances 
                  who are highly likely to switch to competing regional airlines.
   - The Action: Trigger an automated onboarding sequence offering small mile milestones 
                 or lounge vouchers to secure corporate loyalty early.
   - Result: Boosts customer retention rates before travelers abandon the brand.

4. Smarter Credit Card Partnerships
   - The Insight: The data mathematically proves that an immense chunk of overall points 
                  are generated via card transaction paths rather than traditional flights.
   - The Action: Present these detailed cluster sizes to bank networks (like Visa or 
                 Mastercard) to visually show how card rewards directly drive behavior.
   - Result: Massive bargaining leverage to sign highly profitable co-branded credit card 
             contracts for the airline.

In Short:
The business benefits because it translates a massive table of confusing numbers 
into 4 distinct, actionable customer personas. This allows the airline to spend 
less, sell more, and ensure long-term passenger loyalty!
'''


