

# A machine learning project to predict phase states of high-entropy alloys using supervised learning techniques. Implements Logistic Regression and Random Forest models to solve a classification problem for alloy phase prediction.

# Key features:
# - Data preprocessing and exploration
# - Model implementation: Logistic Regression and Random Forest
# - Performance comparison and evaluation
# - Predictive analysis for high-entropy alloy phase states

# Tools: Python, scikit-learn, pandas, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pd.set_option("display.max_columns", 200)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
import torch


%load_ext autoreload
%autoreload 2
%matplotlib inline

HEA_data = pd.read_csv("/content/HEA_Phase_Data.csv")

HEA_data.hist(layout = (4, 3), figsize=(7, 7), bins = 10)   # figsize(w, h) where w = width and h = height

plt.subplots_adjust(wspace=0.5, hspace=1.0)  # wspace = Horizontal spacing between subplots
                                             # hspace = Vertical spacing between subplots
plt.show()

HEA_data.head(10)

HEA_data.tail(10)

HEA_data.columns

HEA_data.dtypes

print(HEA_data.to_string())

HEA_data.shape

HEA_data.shape[0]

HEA_data.shape[1]

HEA_data = HEA_data.drop("References", axis = 1)
HEA_data = HEA_data.drop("Quench_Proc", axis = 1)

HEA_data.describe()

pd.isna(HEA_data).sum()

HEA_data.duplicated()

HEA_data.duplicated().sum()

HEA_data["Alloy_ID"].duplicated().sum()

def nullCount():
  for i in HEA_data.keys():
    if HEA_data[i].isna().sum():
      null_percent = (HEA_data[i].isna().sum()/HEA_data.shape[0]) * 100
      print("{} has a null percent count of {:.1F}%".format(i, (null_percent)))

nullCount()

# Before duplicates were dropped

dup_list = [i for i in HEA_data.keys() if HEA_data[i].duplicated().sum()]
dup_list

len(dup_list)

# NEW DATA FRAME WITH DUPLICATES DROPPED FROM "ALLOY" COLUMN

new_HEA_data = HEA_data.drop_duplicates(subset=['Alloy'])

new_HEA_data

new_HEA_data.hist(layout = (4, 3), figsize = (8, 6), bins = 10)   # figsize(w, h) where w = width and h = height

plt.subplots_adjust(wspace=0.5, hspace=1.0)  # wspace = Horizontal spacing between subplots
                                             # hspace = Vertical spacing between subplots
plt.show()

new_HEA_data.describe()

new_HEA_data.hist(layout = (4, 3), figsize = (8, 6), bins = 10)   # figsize(w, h) where w = width and h = height

plt.subplots_adjust(wspace=0.5, hspace=1.0)  # wspace = Horizontal spacing between subplots
                                             # hspace = Vertical spacing between subplots
plt.show()

def nullCount():
  for i in new_HEA_data.keys():
    if new_HEA_data[i].isna().sum():
      null_percent = (new_HEA_data[i].isna().sum()/new_HEA_data.shape[0]) * 100
      print("{} has a null percent count of {:.2F}%".format(i, (null_percent)))

nullCount()

# After duplicate Alloys (Composition) was dropped

new_HEA_data.nunique()

new_HEA_data["Alloy"].value_counts()

new_HEA_data.isnull().sum()

new_HEA_data.info()

new_HEA_data.isna()

print(new_HEA_data["Alloy"].value_counts())
print()
print(new_HEA_data["Sythesis_Route"].value_counts())
print()
print(new_HEA_data["Phases"].value_counts())


# Original Dataset
HEA_data_num_features = new_HEA_data.select_dtypes(exclude = "object")
HEA_data_num_features.head(10)

HEA_data_cat_features = new_HEA_data.select_dtypes(include = "object")
HEA_data_cat_features.head(10)

HEA_data_num_features.shape

HEA_data_cat_features.shape

new_HEA_data.columns
# the same as new_HEA_data.keys()

new_HEA_data.iloc[0].values

def count_duplicates(column):
    return column.duplicated().sum()


duplicate_counts = new_HEA_data.apply(count_duplicates)
duplicate_counts

def count_duplicate_percentage(column):  #length of the column is the length of the entire dataframe (509)
  duplicate_percent = round((column.duplicated().sum() / len(column)) * 100, 2)
  return duplicate_percent

duplicate_percentages = new_HEA_data.apply(count_duplicate_percentage)  # This code applies the function "count_duplicate_percentage" to every column in the HEA_data DATAFRAME.
duplicate_percentages

# Set the style
sns.set(style="darkgrid")

# Create the subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))

# First subplot
g1 = sns.countplot(x = "Phases", data = new_HEA_data, ax = ax1)
labels1 = [item.get_text() for item in ax1.get_xticklabels()]
ax1.set_xticks(np.arange(len(labels1)))
ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 90, ha = "right")
ax1.set_title("Phases Count")

# Second subplot
g2 = sns.countplot(x = "Sythesis_Route", data = new_HEA_data, ax = ax2)
labels2 = [item.get_text() for item in ax2.get_xticklabels()]
ax2.set_xticks(np.arange(len(labels2)))
ax2.set_xticklabels(ax2.get_xticklabels(), rotation = 90, ha = "right")
ax2.set_title("Synthesis Route Count")


# Adjust layout and show plot
plt.tight_layout()
plt.show()

HEA_data_num_features.corr()

###LABEL ENCODING: CONVERT FROM CATEGORICAL VARIABLE TO NUMERICAL VARIABLE
label_enc = LabelEncoder()

HEA_data_cat_features["Sythesis_Route"] = label_enc.fit_transform(HEA_data_cat_features["Sythesis_Route"])

HEA_data_cat_features.tail(20)

# VARIATIONAL AUTO-ENCODING FOR ARTIFICIAL DATA GENERATION TO FILL IN MISSING VALUES IN DATASET

# DIVIDE DATASET INTO X AND y
#Identifiers = HEA_data[['Alloy_ID', 'Alloy']
#X = HEA_data.drop(['Alloy_ID'. 'Alloy', 'Phases'], axis = 1)
#y = HEA_data['Phases']

# CARRY OUT ONE-HOT ENCODING ON CATEGORICAL ATTRIBUTES IN X AND y
# Step 1: change all categorical features in X to numerical features
# numeric_features = []
# categorical_features = []
# for column in X.columns:
#   if X[column].dtype in ['int64', 'float64']:
#       numeric_features.append(column)
#   elif X[column].dtype == 'object':
#       categorical_features.append(column)
# ____________________________________________
# preprocessor = ColumnTransformer(transformers = [('num', StandardScaler(), numeric_features), ('cat', OneHotEncoder(drop = 'first', sparse = False), categorical_features)])
# _____________________________________________________________________________________________________________________________________________________________________________
# Step 2: change y(categorical variable) to numerical encoded variable
# label_encoder = LabelEncoder()
# y_encoded_var = label_encoder.fit_transform(y)
# print(label_encoder.classes_)
'''Labels for Each Phase structure => FCC = 0, BCC = 1, FCC + BCC = 2, IM = 3'''

# TRAINING and TESTING DATASET SPLIT
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded_var, test_size = 0.2, train_size = 0.8, random_state = 42)

# BUILD AND TRAIN LOGISTICS REGRESSION MODEL WITH TRAIN DATA SET, THEN TEST MODEL PERFORMANCE
#log_reg = LogisticRegression(multi_class = 'multinomial')
#log_reg.fit(X_train, y_train)  -Fits the model
#y_pred = log_reg.predict(X_test)  -To make predictions
# y_pred_original = label_encoder.inverse_transform(y_pred)
# print(y_pred_original)
#log_reg.predict_proba(X_test)
#log_reg.score(X_test, y_test)
#accuracy = accuracy_score(y_test, y_pred)



# BUILD RANDOM FOREST MODEL WITH TRAIN DATA SET AND DECISION TREES, THEN TEST MODEL PERFORMANCE
# rf_classifier = RandomForestClassifier()
# rf_classifier.fit(X_train, y_train)    -Fits the model
# y_pred2 = rf_classifier.predict(X_test)  -To make predictions
# y_pred_original2 = label_encoder.inverse_transform(y_pred2)
# print(y_pred_original2)
# rf_classifier.predict_proba(X_test)
# rf_classifier.score(X_test, y_test)
# accuracy2 = accuracy_score(y_test, y_pred)

# CLASSIFICATIOM REPORT
# classification_report(y_test, y_pred)  -Logistic Regression classification report
# Features = pd.DataFrame(log_reg.feature_importances_index = X.columns)

# classification_report(y_test, y_pred2)  -Random Forest Classification report
# Features = pd.DataFrame(log_reg.feature_importances_index = X.columns)


# USE CROSS VALIDATION TO DETERMINE THE MODEL WITH BEST ACCURACY
# divide training data into k-number of folds
# # use some folds to train and others to validate

# Cross Validate Logistic Regression Model
# cross_validate(log_reg, X, y, cv = 5, scoring = 'accuracy', return_train_score = False)
#                      OR
# cross_val_score(log_reg, X, y, cv = 5)

# Cross Validate Random Forest Model
# cross_validate(rf_classifier, X, y, cv = 5, scoring = 'accuracy', return_train_size = False)
#                     OR
# cross_val_score(rf_classifier, X, y, cv = 5)


