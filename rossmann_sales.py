import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

##Load the data
ross_df = pd.read_csv('train.csv')
store_df = pd.read_csv('store.csv')
test_df = pd.read_csv('test.csv')
submission_df = pd.read_csv('sample_submission.csv')

merged_train_df = ross_df.merge(store_df, how='left', on='Store')
merged_test_df = test_df.merge(store_df, how='left', on='Store')

'''
We can perform the Exploratory Data Analysis (EDA) and study the distribution and relationship between the features with target 
column 'Sales' 
'''

##Preprocessing and Feature Engineering
merged_train_df.info()
#Convert Date to a datecolumn and extract year, month, day and week of year
def split_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week

split_date(merged_train_df)
split_date(merged_test_df)

#We notice that the sales are Zeroes when stores are closed. So we can remove them from the training data
print(merged_train_df[merged_train_df['Open'] == 0]['Sales'].value_counts()) #172817 rows
merged_train_df = merged_train_df[merged_train_df['Open'] == 1].copy()

#Competition -> CompetitionOpenSince[Month/Year] can be computed to number of months
def comp_months(df):
    df['CompetitionOpen'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + (df['Month'] - df['CompetitionOpenSinceMonth'])
    df['CompetitionOpen'] = df['CompetitionOpen'].map(lambda x: 0 if x < 0 else x).fillna(0)

comp_months(merged_train_df)
comp_months(merged_test_df)

#Promotion -> Indicate how long stores has been running 'Promo2' and whether a new round of Promo2 starts in current month
def check_promo_month(df):
    month2str = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 
                 11: 'Nov', 12: 'Dec'}
    try:
        months = (row['PromoInterval'] or '').split(',')
        if row['Promo2Open'] and month2str[row['Month']] in months:
            return 1
        else:
            return 0
    except Exception:
        return 0

def promo_cols(df):
    #Months since Promo2 started
    df['Promo2Open'] = 12 * (df['Year'] - df['Promo2SinceYear']) + (df['WeekOfYear'] - df['Promo2SinceWeek']) * (12 / 52)
    df['Promo2Open'] = df['Promo2Open'].map(lambda x: 0 if x < 0 else x).fillna(0) * df['Promo2']
    #Whether a new round of Promo2 starts in current month
    df['IsPromo2Month'] = df.apply(check_promo_month, axis=1) * df['Promo2']

promo_cols(merged_train_df)
promo_cols(merged_test_df)
# print(merged_train_df[(merged_train_df['Store'] == 825) & (merged_train_df['Date'] == '2015-03-03')])
# print(merged_train_df.loc[168074])

##Input and Target columns
merged_train_df.columns
input_cols = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance', 
              'CompetitionOpen', 'Year', 'Month', 'Day', 'WeekOfYear', 'Promo2', 'Promo2Open', 'IsPromo2Month']
target_cols = 'Sales'

inputs = merged_train_df[input_cols].copy()
targets = merged_train_df[target_cols].copy()
test_inputs = merged_test_df[input_cols].copy()

numeric_cols = inputs.select_dtypes(include=np.number).columns.tolist()
numeric_cols.remove('DayOfWeek')
categorical_cols = inputs.select_dtypes('object').columns.tolist()
categorical_cols.insert(0, 'DayOfWeek')
print(f"Numeric cols: {numeric_cols}\nCategorical cols: {categorical_cols}")

