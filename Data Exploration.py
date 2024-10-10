# Databricks notebook source
# MAGIC %pip install openpyxl
# MAGIC
# MAGIC # Import libraries
# MAGIC import pandas as pd
# MAGIC import matplotlib.pyplot as plt
# MAGIC import seaborn as sns
# MAGIC import numpy as np

# COMMAND ----------

#import data
dbutils.fs.ls("/FileStore/tables/")

# COMMAND ----------

df = pd.ExcelFile('/dbfs/FileStore/tables/Indo.xlsx', engine='openpyxl')

# COMMAND ----------

sheet_df = df.parse('Data Dump - Cleaned')
display(sheet_df)

# COMMAND ----------

sheet_df.columns

# COMMAND ----------

sheet_df.info()

# COMMAND ----------

df_use = sheet_df.copy()
df_use

# COMMAND ----------

df_use[df_use['ListPrice'].isnull()]

# COMMAND ----------

df_use.replace("NaN", 0, inplace=True)

# COMMAND ----------

df_use['PlannedPromoSpend'] = df_use['PlannedPromoSpend'].astype('float64')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Exploration - Categorical

# COMMAND ----------

cat_column = ['Customer', 'PromoIDText', 'PromoGroup', 'PromoFlag', 'PromotionStatus',
       'PromoMechanic', 'PromoShopperMechanic', 'Level1Name_PPH', 'Category',
       'Level4Name_PPH', 'Brand', 'Level6Name_PPH', 'CPG', 'SPF',
       'ProductNameSku_PPH', 'fundtype'

]

df_cat = df_use[cat_column]
df_cat.info()

# COMMAND ----------

custumer_counts = df_cat['Customer'].value_counts()
custumer_counts.plot(kind='bar')

# COMMAND ----------

custumer_counts.describe()

# COMMAND ----------

# # bin as a new feature

# df_cat['Customer_bin'] = pd.cut((df_cat['Customer'].map(df_use['Customer']).value_counts()), 
#         bins=[0,100,1000,10000,float('inf')], 
#         labels=['very low', 'low', 'medium', 'high'])


# COMMAND ----------

PromoIDText_counts = df_cat['PromoIDText'].value_counts()
PromoIDText_counts.describe()

# too many different promoID's

# COMMAND ----------

PromoGroup_counts = df_cat['PromoGroup'].value_counts()
PromoGroup_counts.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature exploration - Numerical

# COMMAND ----------

num_column = [
        'ListPrice',
       'PlannedPromoSalesValueSellIn', 'PlannedPromoSalesVolumeSellIn',
       'PlannedNetPromoGSVSellIn', 'PlannedNetPromoNIVSellIn',
       'PlannedTTSOnSpend', 'PlannedNetPromoTOSellIn', 'PlannedTTSOffSpend',
       'PlannedNetPromoGrossProfitsSellIn', 'PlannedNetPromoCOGSSellIn',
       'PlannedNetPromoPBOSellIn', 'PlannedBaselineValue',
       'PlannedBaselineVolume', 'PlannedBaseGSVSellIn',
       'PlannedBaseTTSOnSpend', 'PlannedBaseNIVSellIn',
       'PlannedBaseTTSOffSpend', 'PlannedBaseTOSellIn',
       'PlannedBaseGrossProfitsSellIn', 'PlannedBaseCOGSSellIn',
       'PlannedBasePBOSellIn', 'PlannedPromoSpend',
       'ActualNetPromoSalesVolumeSellOut', 'ActualBaselineValue',
       'ActualBaselineVolume']

df_num = df_use[num_column]
df_num


# COMMAND ----------

stats = df_num['ActualNetPromoSalesVolumeSellOut'].describe()
stats_df = pd.DataFrame(stats)
pd.options.display.float_format = '{:,.2f}'.format
print(stats_df)

# COMMAND ----------

skewness = df_num['ActualNetPromoSalesVolumeSellOut'].skew()
print(f"Skewness: {skewness}")

# If skewness > 1:

# The sales volume distribution has a long tail on the high end.
# There are likely some promotional periods with exceptionally high sales volumes.


# If skewness < -1:

# The sales volume distribution has a long tail on the low end.
# There might be some promotional periods with unusually low sales volumes.


# If -0.5 < skewness < 0.5:

# The sales volumes are fairly symmetrically distributed.


# If 0.5 < skewness < 1 or -1 < skewness < -0.5:

# The sales volumes are moderately skewed, but not to an extreme degree.

# COMMAND ----------


# deal with negative values and outliers 
df_new_num = df_num[df_num['ActualNetPromoSalesVolumeSellOut'] >= 0]
skewness_ver1 = df_new_num.skew()
print(f"Skewness: {skewness_ver1}")

# COMMAND ----------

stats2 = df_num['ActualBaselineVolume'].describe()
stats_df2 = pd.DataFrame(stats2)
pd.options.display.float_format = '{:,.2f}'.format
print(stats_df2)

# COMMAND ----------

stats3 = df_num['PlannedBaselineVolume'].describe()
stats_df3 = pd.DataFrame(stats3)
pd.options.display.float_format = '{:,.2f}'.format
print(stats_df3)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Time series analysis

# COMMAND ----------

def fix_week(x):
    x_str = str(x)
    parts = x_str.split('.') # parts is list
    
    if len(parts[1]) == 1:
        parts[1] = parts[1] + '0'
    
    year = str(parts[0])
    week = str(parts[1])
                 
    return f"{year + '.' + week}" # return a string

# COMMAND ----------

df_time = df_use[['WeekSkID', 'ActualNetPromoSalesVolumeSellOut']]
df_time = df_time[df_time['ActualNetPromoSalesVolumeSellOut'] >= 0]
# df_time['WeekSkID'] = df_time['WeekSkID'].apply(fix_week)

# df_time['WeekSkID'] = df_use['WeekSkID'].astype(str)
df_time['WeekSkID'] = df_use['WeekSkID'].astype(str).apply(lambda x: x + '0' if len(x) == 6 else x)
df_time['Date'] = pd.to_datetime(df_time['WeekSkID'].str[:4] + 
                                 df_time['WeekSkID'].str[5:].str.zfill(2) + 
                                 '1', format='%Y%W%w')
# weekID_use = weekID.apply(fix_week)
# # weekID_use = pd.to_datetime(weekID_use)
# target_vol = df_use['ActualNetPromoSalesVolumeSellOut']

# COMMAND ----------

df_time.head(100)

# COMMAND ----------

plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='ActualNetPromoSalesVolumeSellOut', data=df_time)
plt.title('Sales Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Sales Volume')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# the shaded area is 95% confidence interval

# COMMAND ----------

for year in df_time['Date'].dt.year.unique():
    df_year = df_time[df_time['Date'].dt.year == year]
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Date', y='ActualNetPromoSalesVolumeSellOut', data=df_year)
    plt.title(f'Sales Volume in {year}')
    plt.xlabel('Date')
    plt.ylabel('Sales Volume')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

df_time.info()

# COMMAND ----------


year = [2021, 2022,2023]

for i in year:
    df_year = df_time[df_time['Date'].dt.year == i]
    df_year['week'] = df_year['Date'].dt.isocalendar().week.astype(int)

    plt.figure(figsize=(12,6))
    sns.lineplot(x='week', y= 'ActualNetPromoSalesVolumeSellOut', data= df_year)
    plt.title(f'weekly sales volume in {i}')
    plt.xlabel('week of the year')
    plt.ylabel('volume')
    plt.xticks(range(1,54), rotation=45, ha='right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# df_count = df_time[df_time['ActualNetPromoSalesVolumeSellOut']>0]['ActualNetPromoSalesVolumeSellOut'].value_counts().reset_index()
df_count = df_time['ActualNetPromoSalesVolumeSellOut'].value_counts().reset_index()
df_count.columns = ['ActualNetPromoSalesVolumeSellOut', 'Count']
# df_count_filtered = df_count[(df_count['Count'] <= 1000) & (df_count['Count'] >= 30)]
df_count_filtered = df_count[(df_count['Count'] >= 10)]

# COMMAND ----------

df_count_filtered

# around 10% of the data is 0

# COMMAND ----------

sns.scatterplot(data=df_count_filtered, x='ActualNetPromoSalesVolumeSellOut', y='Count')
sns.set_style("whitegrid")
plt.figure(figsize=(12,6))

plt.tight_layout()
plt.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

df_new_num['ActualNetPromoSalesVolumeSellOut_log'] = np.log1p(df_new_num['ActualNetPromoSalesVolumeSellOut'])
skewness_log = df_new_num['ActualNetPromoSalesVolumeSellOut_log'].skew()
print(f"Skewness: {skewness_log}")

# COMMAND ----------

correlation = df_num.corr()
correlation

# COMMAND ----------


sns.heatmap(correlation, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.figure(figsize=(50,30))

# COMMAND ----------

# pick out the pairs that are highly correlated (threshold >0.8)
high_cor_pairs = []
for i in range(len(correlation.columns)):
    for j in range(i+1, len(correlation.columns)):
        if abs(correlation.iloc[i, j]) > 0.8:
            high_cor_pairs.append((correlation.columns[i], correlation.columns[j], correlation.iloc[i,j]))



# COMMAND ----------

high_cor_pairs

# COMMAND ----------


correlation['ActualNetPromoSalesVolumeSellOut'].sort_values()

# COMMAND ----------

# list out the pairs that contain list price, ActualBaselineValue, PlannedBaseTTSOffSpend 

listout = []
for pairs in high_cor_pairs:
    if ( 
    'ListPrice' in pairs[0] or 'ListPrice' in pairs[1] or
    'ActualBaselineValue' in pairs[0] or 'ActualBaselineValue' in pairs[1] or
    'PlannedBaseTTSOffSpend' in pairs[0] or 'PlannedBaseTTSOffSpend' in pairs[1] or
    'PlannedTTSOffSpend' in pairs[0] or 'PlannedTTSOffSpend' in pairs[1] or
    'PlannedBasePBOSellIn' in pairs[0] or 'PlannedBasePBOSellIn' in pairs[1] or
    'PlannedBaseGrossProfitsSellIn' in pairs[0] or 'PlannedBaseGrossProfitsSellIn' in pairs[1] or
    'PlannedBaseTTSOnSpend' in pairs[0] or 'PlannedBaseTTSOnSpend' in pairs[1] or
    'PlannedBaseCOGSSellIn' in pairs[0] or 'PlannedBaseCOGSSellIn' in pairs[1] or
    'PlannedBaselineValue' in pairs[0] or 'PlannedBaselineValue' in pairs[1] or
    'PlannedBaseNIVSellIn' in pairs[0] or 'PlannedBaseNIVSellIn' in pairs[1] or
    'PlannedBaseGSVSellIn' in pairs[0] or 'PlannedBaseGSVSellIn' in pairs[1] or
    'PlannedBaseTOSellIn' in pairs[0] or 'PlannedBaseTOSellIn' in pairs[1] or
    'PlannedNetPromoNIVSellIn' in pairs[0] or 'PlannedNetPromoNIVSellIn' in pairs[1] or
    'PlannedNetPromoGSVSellIn' in pairs[0] or 'PlannedNetPromoGSVSellIn' in pairs[1]
    ):
        listout.append(pairs)

# COMMAND ----------

listout

# COMMAND ----------

listout_column = ['PlannedBasePBOSellIn', 'PlannedTTSOffSpend', 'PlannedNetPromoGSVSellIn', 'PlannedNetPromoNIVSellIn', 'PlannedNetPromoTOSellIn', 'PlannedNetPromoCOGSSellIn', 'PlannedNetPromoGrossProfitsSellIn','PlannedNetPromoPBOSellIn', 'PlannedBaseGSVSellIn', 'PlannedBaseNIVSellIn', 'PlannedBaseTOSellIn', 'PlannedBaseGrossProfitsSellIn', 'PlannedBaseCOGSSellIn', 'PlannedBaseTTSOnSpend'  ]
df_num_filtered = df_num.drop(columns=listout_column)
corr2 = df_num_filtered.corr()
corr2

# COMMAND ----------

sns.heatmap(corr2, cmap='coolwarm')
plt.figure(figsize=(50,30))

# COMMAND ----------

df_num_filtered.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## 
