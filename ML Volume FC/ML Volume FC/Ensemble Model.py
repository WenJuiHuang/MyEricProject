# Databricks notebook source
# MAGIC %pip install openpyxl

# COMMAND ----------

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dbutils.fs.ls("/FileStore/tables/")
df = pd.ExcelFile('/dbfs/FileStore/tables/Indo.xlsx', engine='openpyxl')
sheet_df = df.parse('Data Dump - Cleaned')
display(sheet_df)



# COMMAND ----------

sheet_df['ListPrice'] = sheet_df['ListPrice'].fillna(0)
sheet_df.info()

# COMMAND ----------

df = sheet_df.copy()

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor, XGBClassifier

# COMMAND ----------

df_new = df[df['ActualNetPromoSalesVolumeSellOut'] >= 0] # this positively huge impacts the model

# creat new feature
df_new['WeekSkID_str'] = df_new['WeekSkID'].astype(str)
df_new['WeekSkID_str'] = df_new['WeekSkID_str'].apply(lambda x: x + '0' if len(x) == 6 else x)
df_new['year'] = df_new['WeekSkID_str'].str[:4]
df_new['year_num'] = df_new['year'].astype(int)
df_new['WeekOftheYear'] = df_new['WeekSkID_str'].str[-2:].astype(str)
df_new['Date'] = pd.to_datetime(df_new['year'] + df_new['WeekOftheYear'] + '1', format='%Y%W%w')
df_new['WeekOftheYear_num'] = df_new['WeekOftheYear'].astype(int)
df_new['WeekoftheMonth'] = df_new['Date'].dt.day.apply(lambda x : (x-1)// 7  + 1).astype(int)


# COMMAND ----------

# Numerical
from sklearn.preprocessing import RobustScaler, PowerTransformer

# Create the transformers
yeo_johnson = PowerTransformer(method='yeo-johnson')
robust_scaler = RobustScaler()

df_new['ActualBaselineVolume_tr'] = yeo_johnson.fit_transform(df_new['ActualBaselineVolume'].values.reshape(-1,1))
df_new['ActualBaselineVolume_tr'] = robust_scaler.fit_transform(df_new['ActualBaselineVolume_tr'].values.reshape(-1,1))

df_new['PlannedBaselineVolume_tr'] = yeo_johnson.fit_transform(df_new['PlannedBaselineVolume'].values.reshape(-1,1))
df_new['PlannedBaselineVolume_tr'] = robust_scaler.fit_transform(df_new['PlannedBaselineVolume_tr'].values.reshape(-1,1))

df_new['PlannedPromoSalesVolumeSellIn_tr'] = yeo_johnson.fit_transform(df_new['PlannedPromoSalesVolumeSellIn'].values.reshape(-1,1))
df_new['PlannedPromoSalesVolumeSellIn_tr'] = robust_scaler.fit_transform(df_new['PlannedPromoSalesVolumeSellIn_tr'].values.reshape(-1,1))

# COMMAND ----------

# Categorical
from sklearn.feature_extraction import FeatureHasher
cat_col = ['PromoIDText', 'ProductNameSku_PPH', 'PromoMechanic', 'Level4Name_PPH']

# hashers = {}
# n_features=16
# for col in cat_col:
#     hasher = FeatureHasher(n_features=n_features, input_type='string')
#     hashed_features = hasher.fit_transform(df_new[[col]].astype(str).values)
#     hashed_array = hashed_features.toarray()

#     hashed_integers = hashed_array.dot(1 << np.arange(hashed_array.shape[1] - 1, -1, -1))

#     # Add new column to the original DataFrame
#     df_new[f'{col}_hash'] = hashed_integers
    
le = LabelEncoder()
# features['PromoIDText'] = le.fit_transform(features['PromoIDText'])
for column in cat_col:
    df_new[column + '_label'] = le.fit_transform(df_new[column])

# COMMAND ----------

df_new

# COMMAND ----------

features = df_new[['year_num', 'WeekOftheYear_num', 'WeekoftheMonth', 'PromoIDText_label', 'ProductNameSku_PPH_label', 'PromoMechanic_label','ActualBaselineVolume_tr', 'PlannedBaselineVolume_tr', 'PlannedPromoSalesVolumeSellIn_tr']]
target = df_new['ActualNetPromoSalesVolumeSellOut']

# COMMAND ----------

features

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model

# COMMAND ----------

# Define the threshold for 'large' values
threshold = np.percentile(target, 75)

#create binary label
y_binary = (target > threshold).astype(int)

# COMMAND ----------

X_train, X_test, y_train, y_test, y_binary_train, y_binary_test = train_test_split( features, target, y_binary, test_size=0.2, random_state=42)

# COMMAND ----------


# create base models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR

# classifier
clf1 = XGBClassifier(random_state=42)
clf2 = RandomForestClassifier(random_state=42)
# clf3 = LogisticRegression(random_state=42)
# clf3 = SVC(probability=True, random_state=42)

classifiers = [
    ('xgb', clf1),
    ('rf', clf2)
    # ('SVC', clf3)
    # ('lr', clf3),
    # ('svc', clf4)
]

reg1 = XGBRegressor(random_state=42)
reg2 = RandomForestRegressor(random_state=42)
reg3 = SVR()

regressors = [
    ('xgb', reg1),
    ('rf', reg2)
    # ('svr', reg3)
]


# COMMAND ----------

# create ensemble model
from sklearn.ensemble import VotingClassifier, VotingRegressor


ensemble_classifier = VotingClassifier(
    estimators=classifiers,
    voting='soft'
)

ensemble_regressor_small = VotingRegressor(estimators=regressors)
ensemble_regressor_large = VotingRegressor(estimators=regressors)

# COMMAND ----------

# train ensemble classifier
ensemble_classifier.fit(X_train, y_binary_train)

# Train ensemble regressors
ensemble_regressor_small.fit(X_train[y_binary_train == 0], y_train[y_binary_train == 0])
ensemble_regressor_large.fit(X_train[y_binary_train == 1], y_train[y_binary_train == 1])

# COMMAND ----------


def predict(X_new_scaled, classifier, regressor_small, regressor_large):
    # X_new_scaled = scaler.transform(X_new)
    
    # Predict probabilities of being in the "large" class
    class_probs = classifier.predict_proba(X_new_scaled)[:, 1]
    
    # Predict values using both regressors
    small_preds = regressor_small.predict(X_new_scaled)
    large_preds = regressor_large.predict(X_new_scaled)
    
    # Weighted average based on class probabilities
    y_pred = (1 - class_probs) * small_preds + class_probs * large_preds
    
    return y_pred

print("Prediction function created.")

# COMMAND ----------

# Make predictions on test set
y_pred = predict(X_test, ensemble_classifier, ensemble_regressor_small, ensemble_regressor_large)

# Evaluate predictions
mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Ensemble model evaluation:")
print(f"Mean Squared Error: {mse:.4f}")
# print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared Score: {r2:.4f}")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


