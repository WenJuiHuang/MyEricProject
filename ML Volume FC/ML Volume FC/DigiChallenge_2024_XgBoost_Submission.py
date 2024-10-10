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

df.info()

# COMMAND ----------


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# COMMAND ----------

df_new = df[df['ActualNetPromoSalesVolumeSellOut'] >= 0] # this positively huge impacts the model


# date 
df_new['WeekSkID_str'] = df_new['WeekSkID'].astype(str)
df_new['WeekSkID_str'] = df_new['WeekSkID_str'].apply(lambda x: x + '0' if len(x) == 6 else x)
df_new['year'] = df_new['WeekSkID_str'].str[:4]
df_new['year_num'] = df_new['year'].astype(int)
df_new['WeekOftheYear'] = df_new['WeekSkID_str'].str[-2:].astype(str)
df_new['Date'] = pd.to_datetime(df_new['year'] + df_new['WeekOftheYear'] + '1', format='%Y%W%w')
df_new['WeekOftheYear_num'] = df_new['WeekOftheYear'].astype(int)
df_new['WeekoftheMonth'] = df_new['Date'].dt.day.apply(lambda x : (x-1)// 7  + 1).astype(int)



# COMMAND ----------

df_new.columns

# COMMAND ----------

# # Numerical
# # create new feature - basic aggregation by CPG
# df_new['PlannedBaselineVolume_max'] = df_new.groupby('CPG')['PlannedBaselineVolume'].transform('max')
# df_new['PlannedBaselineVolume_min'] = df_new.groupby('CPG')['PlannedBaselineVolume'].transform('min')
# df_new['PlannedBaselineVolume_std'] = df_new.groupby('CPG')['PlannedBaselineVolume'].transform('std')
# df_new['PlannedBaselineVolume_mean'] = df_new.groupby('CPG')['PlannedBaselineVolume'].transform('mean')

# COMMAND ----------

df_new

# COMMAND ----------

# Numerical
from sklearn.preprocessing import RobustScaler, PowerTransformer

# Create the transformers
yeo_johnson = PowerTransformer(method='yeo-johnson')
robust_scaler = RobustScaler()

col_to_transform = [
    'ActualBaselineVolume',
    'PlannedBaselineVolume',
    'PlannedPromoSalesVolumeSellIn'
    # 'PlannedBaselineVolume_max',
    # 'PlannedBaselineVolume_min',
    # 'PlannedBaselineVolume_std',
    # 'PlannedBaselineVolume_mean'

]

for col in col_to_transform:
    df_new[f'{col}_tr'] = yeo_johnson.fit_transform(df_new[col].values.reshape(-1,1))
    df_new[f'{col}_tr'] = robust_scaler.fit_transform(df_new[col].values.reshape(-1,1))




# COMMAND ----------

# Categorical
# from sklearn.feature_extraction import FeatureHasher
cat_col = ['PromoIDText', 'PromoMechanic', 'Level1Name_PPH', 'Category', 'Level4Name_PPH', 'Brand', 'Level6Name_PPH', 'CPG', 'SPF', 'PromoFlag']

    
le = LabelEncoder()
all_mappings = {}

for column in cat_col:
    df_new[column + '_label'] = le.fit_transform(df_new[column])

    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    all_mappings[column] = mapping

# COMMAND ----------

# all_mappings

# COMMAND ----------

df_new.columns.tolist()

# COMMAND ----------

# features = df_new[['year_num', 'WeekOftheYear_num', 'WeekoftheMonth', 'PromoIDText_label', 'ProductNameSku_PPH_label', 'PromoMechanic_label','ActualBaselineVolume_tr', 'PlannedBaselineVolume_tr', 'PlannedPromoSalesVolumeSellIn_tr','PlannedBaselineVolume_max_tr', 'PlannedBaselineVolume_min_tr', 'PlannedBaselineVolume_std_tr', 'PlannedBaselineVolume_mean_tr']]
features_df = df_new[['year_num', 'WeekOftheYear_num', 'WeekoftheMonth', 'PromoIDText_label', 'PromoMechanic_label', 'Level1Name_PPH_label', 'Category_label',
       'Level4Name_PPH_label', 'Brand_label', 'Level6Name_PPH_label',
       'CPG_label', 'PromoFlag_label', 'PlannedBaselineVolume_tr',
 'PlannedPromoSalesVolumeSellIn_tr', 'ProductNameSku_PPH', 'SPF_label' ]]
target = df_new['ActualNetPromoSalesVolumeSellOut']

# COMMAND ----------

features_df

# COMMAND ----------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
import joblib

def fit_tfidf_and_kmeans(df, name_column='ProductNameSku_PPH', max_features=400, ngram_range=(1, 2), n_clusters=10):
    unique_products = df[name_column].unique()

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=2, max_df=0.95)
    tfidf_matrix = tfidf.fit_transform(unique_products)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)

    # Save the fitted objects
    joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
    joblib.dump(kmeans, 'kmeans_model.joblib')

    return tfidf, kmeans

def transform_tfidf_and_kmeans(df, tfidf, kmeans, name_column='ProductNameSku_PPH'):
    unique_products = df[name_column].unique()

    # Transform using the fitted TF-IDF vectorizer
    tfidf_matrix = tfidf.transform(unique_products)

    # Convert TF-IDF matrix to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{col}' for col in tfidf.get_feature_names_out()])

    # Predict clusters using the fitted KMeans model
    cluster_labels = kmeans.predict(tfidf_matrix)

    # Add cluster labels and original product names to the DataFrame
    tfidf_df['cluster'] = cluster_labels
    tfidf_df[name_column] = unique_products

    return tfidf_df

# COMMAND ----------



# COMMAND ----------


tfidf, kmeans = fit_tfidf_and_kmeans(features_df)
train_tfidf_df = transform_tfidf_and_kmeans(features_df, tfidf, kmeans)

# COMMAND ----------

train_tfidf_df

# COMMAND ----------


features = features_df.merge(train_tfidf_df, on='ProductNameSku_PPH', how='left')
features = features.drop('ProductNameSku_PPH', axis=1)

# COMMAND ----------

features

# COMMAND ----------

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesRegressor
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

xgb_model = XGBRegressor(random_state=42, n_estimators= 100)
xgb_model.fit(X_train, y_train)

val_pred1 = xgb_model.predict(X_val)
X_val_level2 = val_pred1.reshape(-1, 1)

et = ExtraTreesRegressor(n_estimators=100, random_state=42)
et.fit(X_val_level2, y_val)
et_pred = et.predict(X_val_level2)



# xgb_model = XGBRegressor(random_state=42, n_estimators= 100)
# xgb_model.fit(X_train, y_train)


# score_train = xgb_model.score(X_train, y_train)
# score_test = xgb_model.score(X_test, y_test)
# print(f"Model Score Train: {score_train}")
# print(f"Model Score Test: {score_test}")


# COMMAND ----------

# # Evaluate the model
# y_pred = xgb_model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)


# print(f"Root Mean Squared Error: {rmse}")
# print(f"MSE: {mse}")

mse = mean_squared_error(y_val, et_pred)
print(f"Final MSE: {mse}")


# COMMAND ----------

X_train.columns.tolist()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation /Submission set

# COMMAND ----------

validation_set = spark.read.table("hive_metastore.default.validation_tpa_id")

validation_set = validation_set.toPandas()

# COMMAND ----------

validation_set

# COMMAND ----------

display(validation_set)

# COMMAND ----------

validation_set.info()

# COMMAND ----------

# date 
validation_set['WeekSkID_str'] = validation_set['WeekSkID'].astype(str)
validation_set['WeekSkID_str'] = validation_set['WeekSkID_str'].apply(lambda x: x + '0' if len(x) == 6 else x)
validation_set['year'] = validation_set['WeekSkID_str'].str[:4]
validation_set['year_num'] = validation_set['year'].astype(int)
validation_set['WeekOftheYear'] = validation_set['WeekSkID_str'].str[-2:].astype(str)
validation_set['Date'] = pd.to_datetime(validation_set['year'] + validation_set['WeekOftheYear'] + '1', format='%Y%W%w')
validation_set['WeekOftheYear_num'] = validation_set['WeekOftheYear'].astype(int)
validation_set['WeekoftheMonth'] = validation_set['Date'].dt.day.apply(lambda x : (x-1)// 7  + 1).astype(int)

# COMMAND ----------

# Numerical
from sklearn.preprocessing import RobustScaler, PowerTransformer

# Create the transformers
yeo_johnson = PowerTransformer(method='yeo-johnson')
robust_scaler = RobustScaler()

col_to_transform = [
    'ActualBaselineVolume',
    'PlannedBaselineVolume',
    'PlannedPromoSalesVolumeSellIn'
    # 'PlannedBaselineVolume_max',
    # 'PlannedBaselineVolume_min',
    # 'PlannedBaselineVolume_std',
    # 'PlannedBaselineVolume_mean'

]

for col in col_to_transform:
    validation_set[f'{col}_tr'] = yeo_johnson.fit_transform(validation_set[col].values.reshape(-1,1))
    validation_set[f'{col}_tr'] = robust_scaler.fit_transform(validation_set[col].values.reshape(-1,1))


# COMMAND ----------

# Categorical

cat_col = ['PromoIDText', 'PromoMechanic', 'Level1Name_PPH', 'Category', 'Level4Name_PPH', 'Brand', 'Level6Name_PPH', 'CPG', 'SPF', 'PromoFlag']

    
le = LabelEncoder()
all_mappings = {}

for column in cat_col:
    validation_set[column + '_label'] = le.fit_transform(validation_set[column])

    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    all_mappings[column] = mapping

# COMMAND ----------

features_val = validation_set[['year_num', 'WeekOftheYear_num', 'WeekoftheMonth', 'PromoIDText_label', 'PromoMechanic_label', 'Level1Name_PPH_label', 'Category_label',
       'Level4Name_PPH_label', 'Brand_label', 'Level6Name_PPH_label',
       'CPG_label', 'PromoFlag_label', 'PlannedBaselineVolume_tr',
 'PlannedPromoSalesVolumeSellIn_tr', 'ProductNameSku_PPH', 'SPF_label'  ]]
# target_val = validation_set['ActualNetPromoSalesVolumeSellOut']

# COMMAND ----------

# tdf_feature = create_tfidf_product_features(features_val)
loaded_tfidf = joblib.load('tfidf_vectorizer.joblib')
loaded_kmeans = joblib.load('kmeans_model.joblib')
tdf_feature = transform_tfidf_and_kmeans(features_val, loaded_tfidf, loaded_kmeans)

# COMMAND ----------

features_val_to_use = features_val.merge(tdf_feature, on='ProductNameSku_PPH', how='left')
features_val_to_use = features_val_to_use.drop('ProductNameSku_PPH', axis=1)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

features_val_to_use

# COMMAND ----------

features_val_to_use.columns.tolist()

# COMMAND ----------

def predict_new_data(X_new):
    # Generate first-level predictions
    pred = xgb_model.predict(X_new)
    
    # Reshape for second-level model input
    X_new_level2 = pred.reshape(-1, 1)
    
    # Generate second-level predictions
    final_pred = et.predict(X_new_level2)
    
    return final_pred

predictions = predict_new_data(features_val_to_use)


# COMMAND ----------

validation_set['ActualNetPromoSalesVolumeSellOut'] = predictions

# COMMAND ----------

# Keep only the specified columns
validation_set = validation_set[['Customer', 'PromoIDText', 'ProductNameSku_PPH', 'WeekSkID', 'ActualNetPromoSalesVolumeSellOut']]

# COMMAND ----------

display(validation_set)

# COMMAND ----------



# COMMAND ----------

# features = df.drop(['ActualNetPromoSalesVolumeSellOut', 'InstoreStartDate', 'InstoreEndDate'], axis=1)
# target = df['ActualNetPromoSalesVolumeSellOut']
# from sklearn.preprocessing import OrdinalEncoder




# percentile_75 = df_new['ActualNetPromoSalesVolumeSellOut'].quantile(0.75)
# df_new = df_new[df_new['ActualNetPromoSalesVolumeSellOut'] <= percentile_75]

# creat new feature
# df_new['PlannedTTSSpend'] = df_new['PlannedTTSOnSpend'] + df_new['PlannedTTSOffSpend']




# encoder = OrdinalEncoder()
# df_new['year_ordinal'] = encoder.fit_transform(df_new[['year']])
# df_new['WeekOftheYear_ordinal'] = encoder.fit_transform(df_new[['WeekOftheYear']])
# df_new['WeekoftheMonth_ordinal'] = encoder.fit_transform(df_new[['WeekoftheMonth']])

#log
# df_new['ActualBaselineVolume_log'] = np.log1p(df_new['ActualBaselineVolume'])
# df_new['PlannedBaselineVolume_log'] = np.log1p(df_new['PlannedBaselineVolume'])
# df_new['PlannedTTSSpend_log'] = np.log1p(df_new['PlannedTTSSpend'])


# df_new['PlannedPromoSalesVolumeSellIn_log'] = np.log1p(df_new['PlannedPromoSalesVolumeSellIn'])
# features = df_new[['year_ordinal', 'WeekOftheYear_ordinal', 'WeekoftheMonth_ordinal', 'PromoIDText', 'Level6Name_PPH', 'ActualBaselineVolume_log', 'PlannedBaselineVolume_log']]
# target = df_new['ActualNetPromoSalesVolumeSellOut']

# customer feature negativly impact the model

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# correlation_value = df_new['ActualNetPromoSalesVolumeSellOut'].corr(df_new['PlannedTTSSpend'])
# print(correlation_value)

# COMMAND ----------

# to_show = df_new['PlannedTTSSpend'].describe()
# pd.options.display.float_format = '{:,.2f}'.format
# print(to_show)

# COMMAND ----------

# features.info()

# COMMAND ----------



# COMMAND ----------

# Encode
# categorical_columns = features.select_dtypes(include=['object']).columns
# numerical_columns = features.select_dtypes(include=['int64', 'float64']).columns

# le = LabelEncoder()
# features['PromoIDText'] = le.fit_transform(features['PromoIDText'])
# for column in features.select_dtypes(include=['object']):
#     features[column] = le.fit_transform(features[column])

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

# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# # Fit the pipeline
# xgb_pipeline.fit(X_train, y_train)

# # Make predictions
# y_pred = xgb_pipeline.predict(X_test)

# xgb_model = XGBRegressor(random_state=42)
# xgb_model.fit(X_train, y_train)

# COMMAND ----------

# xgb_pipeline.fit(X_train, y_train)
# score_train = xgb_model.score(X_train, y_train)
# score = xgb_model.score(X_test, y_test)
# print(f"Model Score Train: {score_train}")
# print(f"Model Score: {score}")

# COMMAND ----------

# # Evaluate the model
# y_pred = xgb_model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# # r2 = r2_score(y_test, y_pred)

# print(f"Root Mean Squared Error: {rmse}")
# print(f"MSE: {mse}")

# COMMAND ----------



# COMMAND ----------

# importance = xgb_model.feature_importances_

# feature_importance_df = pd.DataFrame({
#     'feature': X_train.columns,
#     'importance': importance
# }).sort_values('importance', ascending=False)

# COMMAND ----------

# feature_importance_df

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


