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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# COMMAND ----------

df_new = df[df['ActualNetPromoSalesVolumeSellOut'] >= 0] # this positively huge impacts the model
# df_new = df.copy()

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

# # numerical to cat
# nu_col = ['ListPrice',
#        'PlannedPromoSalesValueSellIn', 'PlannedPromoSalesVolumeSellIn',
#        'PlannedNetPromoGSVSellIn', 'PlannedNetPromoNIVSellIn',
#        'PlannedTTSOnSpend', 'PlannedNetPromoTOSellIn', 'PlannedTTSOffSpend',
#        'PlannedNetPromoGrossProfitsSellIn', 'PlannedNetPromoCOGSSellIn',
#        'PlannedNetPromoPBOSellIn', 'PlannedBaselineValue',
#        'PlannedBaselineVolume', 'PlannedBaseGSVSellIn',
#        'PlannedBaseTTSOnSpend', 'PlannedBaseNIVSellIn',
#        'PlannedBaseTTSOffSpend', 'PlannedBaseTOSellIn',
#        'PlannedBaseGrossProfitsSellIn', 'PlannedBaseCOGSSellIn',
#        'PlannedBasePBOSellIn']
       
# for nu in nu_col:
#     df_new[nu + '_str'] = df_new[nu].astype(str)

# # df_new['ListPrice_str'] = df_new['ListPrice'].astype(str)
# # df_new['PlannedPromoSalesValueSellIn_str'] = df_new['PlannedPromoSalesValueSellIn'].astype(str)

# COMMAND ----------

df_new.columns.tolist()

# COMMAND ----------

# Categorical
# from sklearn.feature_extraction import FeatureHasher
cat_col = ['PromoIDText', 'PromoMechanic', 'Level1Name_PPH', 'Category', 'Level4Name_PPH', 'Brand', 'Level6Name_PPH', 'CPG', 'SPF', 'PromoFlag']

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
all_mappings = {}
# features['PromoIDText'] = le.fit_transform(features['PromoIDText'])
for column in cat_col:
    df_new[column + '_label'] = le.fit_transform(df_new[column])

    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    all_mappings[column] = mapping

#     'ListPrice_str',
#        'PlannedPromoSalesValueSellIn_str',
#        'PlannedPromoSalesVolumeSellIn_str', 'PlannedNetPromoGSVSellIn_str',
#        'PlannedNetPromoNIVSellIn_str', 'PlannedTTSOnSpend_str',
#        'PlannedNetPromoTOSellIn_str', 'PlannedTTSOffSpend_str',
#        'PlannedNetPromoGrossProfitsSellIn_str',
#        'PlannedNetPromoCOGSSellIn_str', 'PlannedNetPromoPBOSellIn_str',
#        'PlannedBaselineValue_str', 'PlannedBaselineVolume_str',
#        'PlannedBaseGSVSellIn_str', 'PlannedBaseTTSOnSpend_str',
#        'PlannedBaseNIVSellIn_str', 'PlannedBaseTTSOffSpend_str',
#        'PlannedBaseTOSellIn_str', 'PlannedBaseGrossProfitsSellIn_str',
#        'PlannedBaseCOGSSellIn_str', 'PlannedBasePBOSellIn_str'

# COMMAND ----------

df_new.columns.tolist()

# COMMAND ----------

features_df = df_new[['year_num', 'WeekOftheYear_num', 'WeekoftheMonth', 'PromoIDText_label', 'PromoMechanic_label', 'Level1Name_PPH_label', 'Category_label',
       'Level4Name_PPH_label', 'Brand_label', 'Level6Name_PPH_label',
       'CPG_label', 'PromoFlag_label', 'PlannedBaselineVolume_tr',
 'PlannedPromoSalesVolumeSellIn_tr', 'ProductNameSku_PPH', 'SPF_label' ]]

# ,'ActualBaselineVolume_tr'
# , 'PlannedBaselineVolume_tr', 'PlannedPromoSalesVolumeSellIn_tr'
# , 'SPF_label'
#  'ListPrice_str_label',
#        'PlannedPromoSalesValueSellIn_str_label',
#  'PlannedNetPromoGSVSellIn_str_label',
#  'PlannedNetPromoNIVSellIn_str_label',
# #  'PlannedNetPromoGrossProfitsSellIn_str_label'

# COMMAND ----------

features_df

# COMMAND ----------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re

def create_tfidf_product_features(df, name_column='ProductNameSku_PPH', max_features=400, ngram_range=(1, 2), n_clusters=10 ):
    unique_products = df[name_column].unique()

    # # Basic cleaning
    # def clean_name(name):
    #     # Remove numbers and special characters, convert to lowercase
    #     return re.sub(r'[^a-zA-Z\s]', '', name).lower()
    
    # df['cleaned_name'] = df[name_column].apply(clean_name)
    # cleaned_names = [clean_name(name) for name in unique_products]
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    # tfidf_matrix = tfidf.fit_transform(df['cleaned_name'])
    tfidf_matrix = tfidf.fit_transform(unique_products)
    
    # Convert TF-IDF matrix to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{col}' for col in tfidf.get_feature_names_out()])

    # # Add the original product names
    # tfidf_df[name_column] = unique_products

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    # Add cluster labels and original product names to the DataFrame
    tfidf_df['cluster'] = cluster_labels
    tfidf_df[name_column] = unique_products

    # Create a mapping of product names to cluster labels
    product_to_cluster = dict(zip(unique_products, cluster_labels))

    # # Add cluster labels to the original DataFrame
    # df['cluster'] = df[name_column].map(product_to_cluster)

    return tfidf_df
   


# COMMAND ----------

tdf_feature = create_tfidf_product_features(features_df)

# COMMAND ----------

tdf_feature

# COMMAND ----------

# features = features_df.join(tdf_feature)
features = features_df.merge(tdf_feature, on='ProductNameSku_PPH', how='left')
# features = features.drop(['ProductNameSku_PPH', 'cleaned_name'], axis=1)
features = features.drop('ProductNameSku_PPH', axis=1)

# COMMAND ----------

features

# COMMAND ----------

features.info()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# features = df_new[['year_num', 'WeekOftheYear_num', 'WeekoftheMonth', 'PromoIDText_label', 'ProductNameSku_PPH_label', 'PromoMechanic_label','ActualBaselineVolume_tr', 'PlannedBaselineVolume_tr', 'PlannedPromoSalesVolumeSellIn_tr']]
target = df_new['ActualNetPromoSalesVolumeSellOut']

# COMMAND ----------

features

# COMMAND ----------

# from sklearn.model_selection import cross_val_score
# from sklearn.ensemble import ExtraTreesRegressor
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

# xgb_model = XGBRegressor(random_state=42, n_estimators= 100)
# # , learning_rate=0.1, max_depth=5, min_child_weight= 3, n_estimators= 200
# xgb_model.fit(X_train, y_train)

# val_pred1 = xgb_model.predict(X_val)
# X_val_level2 = val_pred1.reshape(-1, 1)

# et = ExtraTreesRegressor(n_estimators=100, random_state=42)
# et.fit(X_val_level2, y_val)
# et_pred = et.predict(X_val_level2)

# # final_pred = 0.6 * et_pred + 0.4 * lr_pred

# # score_train = xgb_model.score(X_train, y_train)
# # score_test = xgb_model.score(X_test, y_test)
# # print(f"Model Score Train: {score_train}")
# # print(f"Model Score Test: {score_test}")


# First, split the data into training+validation and test sets
X_temp, X_test, y_temp, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Then split the temp data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_nn_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


model = create_nn_model(X_train.shape[1])
history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val), 
                    epochs=100, 
                    batch_size=32,
                    verbose=1)


# COMMAND ----------

# Function to evaluate the model
def evaluate_model(model, X, y, dataset_name):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    rmse = np.sqrt(mse)
    
    print(f"{dataset_name} set evaluation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    print()

# COMMAND ----------

evaluate_model(model, X_train, y_train, "Training")

# Evaluate on validation set
evaluate_model(model, X_val, y_val, "Validation")

# Evaluate on test set
evaluate_model(model, X_test, y_test, "Test")

# COMMAND ----------

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# # # # Evaluate the model
# # y_pred = xgb_model.predict(X_test)
# # mse = mean_squared_error(y_test, y_pred)
# # rmse = np.sqrt(mse)


# # print(f"Root Mean Squared Error: {rmse}")
# # print(f"MSE: {mse}")
# mse = mean_squared_error(y_val, et_pred)
# print(f"Final MSE: {mse}")


# COMMAND ----------

# features_rev = features.copy()
# cat_rev = ['PromoMechanic_label', 'ProductNameSku_PPH_label' ]


# for col in cat_rev:
#     re = LabelEncoder()
#     re.fit(features_rev[col])

#     features_rev[column + '_ori'] = re.inverse_transform(features_rev[col])

# COMMAND ----------

# features_rev


# COMMAND ----------



# COMMAND ----------


# # grid search
# from sklearn.model_selection import GridSearchCV
# # from sklearn.ensemble import GradientBoostingRegressor

# param_grid={
#     'max_depth': [3,5,7],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [100, 200, 300],
#     'min_child_weight': [3, 5, 7],
#     'subsample': [0.7, 0.8, 0.9],
#     'colsample_bytree': [0.7, 0.8, 0.9],
#     'gamma': [0, 0.1, 0.2]
# }

# xgb = XGBRegressor(
#     random_state=42,
#     early_stopping_rounds=10,
#     eval_metric=mean_squared_error
#     )

# grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, verbose=2)
# grid_search.fit(X_train, y_train, eval_set=[(X_test, y_test)])

# print("Best parameters:", grid_search.best_params_)
# print("Best cross-validation score:", grid_search.best_score_)

# # Use the best model
# best_model = grid_search.best_estimator_
# y_pred_1 = best_model.predict(X_test)




# COMMAND ----------

# Evaluate the best model
# mse_1 = mean_squared_error(y_test, y_pred_1)
# rmse_1 = np.sqrt(mse)
# print(f"Root Mean Squared Error: {rmse_1}")
# print(f"MSE: {mse_1}")

# COMMAND ----------

df_to_see = X_val.copy()
df_to_see['ActualNetPromoSalesVolumeSellOut'] = et_pred
df_to_see['Ans'] = y_val

# COMMAND ----------

df_to_see.columns.tolist()

# COMMAND ----------

y_test

# COMMAND ----------

display(df_to_see)

# COMMAND ----------

importance = xgb_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importance
}).sort_values('importance', ascending=False)

# COMMAND ----------

feature_importance_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

validation_set = spark.read.table("hive_metastore.default.validation_tpa_id")

validation_set = validation_set.toPandas()

# COMMAND ----------

validation_set

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



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


