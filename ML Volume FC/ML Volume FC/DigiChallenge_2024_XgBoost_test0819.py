# Databricks notebook source
# MAGIC %pip install openpyxl

# COMMAND ----------

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# COMMAND ----------

dbutils.fs.ls("/FileStore/tables/")
df = pd.ExcelFile('/dbfs/FileStore/tables/Indo.xlsx', engine='openpyxl')
sheet_df = df.parse('Data Dump - Cleaned')
display(sheet_df)

# COMMAND ----------

sheet_df.columns

# COMMAND ----------


sheet_df['ListPrice'] = sheet_df['ListPrice'].fillna(0)

# COMMAND ----------


sheet_df.info()

# COMMAND ----------

df = sheet_df.copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessing

# COMMAND ----------

# drop columns

# df.drop(['InStoreStartWeek', 'InStoreEndWeek' , 'PromoFlag', 'PromotionStatus', 'ShipmentStartDate', 'ShipmentEndDate', 'PromoShopperMechanic', 'ProductNameSku_PPH', 'PlannedPromoSpend'], axis=1, inplace=True)

# COMMAND ----------

# # fix week porblem

# def fix_week(x):
#     x_str = str(x)
#     parts = x_str.split('.') # parts is list
    
#     if len(parts[1]) == 1:
#         parts[1] = parts[1] + '0'
    
#     year = str(parts[0])
#     week = str(parts[1])
                 
#     return f"{year + '.' + week}" # return a string

# COMMAND ----------

# df['WeekSkID'] = df['WeekSkID'].apply(fix_week)

# COMMAND ----------

df.info()

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# COMMAND ----------

# features = df.drop(['ActualNetPromoSalesVolumeSellOut', 'InstoreStartDate', 'InstoreEndDate'], axis=1)
# target = df['ActualNetPromoSalesVolumeSellOut']
from sklearn.preprocessing import OrdinalEncoder


df_new = df[df['ActualNetPromoSalesVolumeSellOut'] >= 0] # this positively huge impacts the model

# percentile_75 = df_new['ActualNetPromoSalesVolumeSellOut'].quantile(0.75)
# df_new = df_new[df_new['ActualNetPromoSalesVolumeSellOut'] <= percentile_75]

# creat new feature
# df_new['PlannedTTSSpend'] = df_new['PlannedTTSOnSpend'] + df_new['PlannedTTSOffSpend']
df_new['WeekSkID'] = df['WeekSkID'].astype(str).apply(lambda x: x + '0' if len(x) == 6 else x)
df_new['year'] = df_new['WeekSkID'].str[:4]
df_new['WeekOftheYear'] = df_new['WeekSkID'].str[-2:]
df_new['Date'] = pd.to_datetime(df_new['year'] + df_new['WeekOftheYear'] + '1', format='%Y%W%w')
df_new['WeekoftheMonth'] = df_new['Date'].dt.day.apply(lambda x : (x-1)// 7  + 1).astype(str)

encoder = OrdinalEncoder()
df_new['year_ordinal'] = encoder.fit_transform(df_new[['year']])
df_new['WeekOftheYear_ordinal'] = encoder.fit_transform(df_new[['WeekOftheYear']])
df_new['WeekoftheMonth_ordinal'] = encoder.fit_transform(df_new[['WeekoftheMonth']])

#log
df_new['ActualBaselineVolume_log'] = np.log1p(df_new['ActualBaselineVolume'])
df_new['PlannedBaselineVolume_log'] = np.log1p(df_new['PlannedBaselineVolume'])
# df_new['PlannedTTSSpend_log'] = np.log1p(df_new['PlannedTTSSpend'])


# df_new['PlannedPromoSalesVolumeSellIn_log'] = np.log1p(df_new['PlannedPromoSalesVolumeSellIn'])
features = df_new[['year_ordinal', 'WeekOftheYear_ordinal', 'WeekoftheMonth_ordinal', 'PromoIDText', 'Level6Name_PPH', 'ActualBaselineVolume_log', 'PlannedBaselineVolume_log']]
target = df_new['ActualNetPromoSalesVolumeSellOut']

# customer feature negativly impact the model

# COMMAND ----------

# correlation_value = df_new['ActualNetPromoSalesVolumeSellOut'].corr(df_new['PlannedTTSSpend'])
# print(correlation_value)

# COMMAND ----------

# to_show = df_new['PlannedTTSSpend'].describe()
# pd.options.display.float_format = '{:,.2f}'.format
# print(to_show)

# COMMAND ----------

features.info()

# COMMAND ----------



# COMMAND ----------

# Encode
# categorical_columns = features.select_dtypes(include=['object']).columns
# numerical_columns = features.select_dtypes(include=['int64', 'float64']).columns

le = LabelEncoder()
# features['PromoIDText'] = le.fit_transform(features['PromoIDText'])
for column in features.select_dtypes(include=['object']):
    features[column] = le.fit_transform(features[column])

# COMMAND ----------

features

# COMMAND ----------



# COMMAND ----------

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', 'passthrough', numerical_columns),
#         ('cat', LabelEncoder, categorical_columns)
#     ])


# xgb = XGBRegressor(random_state=42)

# COMMAND ----------

# xgb_pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('regressor', xgb)
# ])

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# # Fit the pipeline
# xgb_pipeline.fit(X_train, y_train)

# # Make predictions
# y_pred = xgb_pipeline.predict(X_test)

xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# COMMAND ----------

# xgb_pipeline.fit(X_train, y_train)
score_train = xgb_model.score(X_train, y_train)
score = xgb_model.score(X_test, y_test)
print(f"Model Score Train: {score_train}")
print(f"Model Score: {score}")

# COMMAND ----------

# Evaluate the model
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: {rmse}")
print(f"MSE: {mse}")

# COMMAND ----------



# COMMAND ----------

importance = xgb_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importance
}).sort_values('importance', ascending=False)

# COMMAND ----------

feature_importance_df

# COMMAND ----------

# y_test_ori = np.expm1(y_test)
# y_pred_ori = np.expm1(y_pred)
# mse = mean_squared_error(y_test_ori, y_pred_ori)
# print(f"MSE: {mse}")

# COMMAND ----------

# xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.1, max_depth=6, random_state=42)

# eval_set = [(X_train, y_train), (X_test, y_test)]
# xgb_model.fit(X_train, y_train, eval_metric="rmse", eval_set=eval_set, verbose=False, early_stopping_rounds=50)

# results = xgb_model.evals_result()

# COMMAND ----------

# epochs = len(results['validation_0']['rmse'])
# x_axis = range(0, epochs)

# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
# ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
# ax.legend()
# plt.ylabel('RMSE')
# plt.title('XGBoost Learning Curve')
# plt.xlabel('Number of trees')

# # Add vertical line at optimal number of trees
# best_iteration = xgb_model.best_iteration
# plt.axvline(x=best_iteration, color='r', linestyle='--')
# plt.text(best_iteration, ax.get_ylim()[1], f'Optimal trees: {best_iteration}', 
#          horizontalalignment='center', verticalalignment='bottom')

# plt.show()

# print(f"Optimal number of trees: {xgb_model.best_iteration}")

# COMMAND ----------


