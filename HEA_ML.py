

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
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix, roc_curve, roc_auc_score
from sklearn.compose import ColumnTransformer
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
import torch



df= pd.read_csv("Sure_DS_Phase_Data - HEA_Phase_Data.csv.csv")
df.info()
df.columns

df.drop(columns=["Alloy_ID","Num_of_Elem","Alloy","Microstructure_","HPR","IM_Structure"],inplace=True)
df2= df.drop_duplicates()

df2_dummies=pd.get_dummies(df2["Hot-Cold_Working Temp"],dtype=float)
df3=pd.concat([df2,df2_dummies],axis=1)
df3.drop(columns="Hot-Cold_Working Temp",inplace=True)

df3_dummies=pd.get_dummies(df3["Quench_Proc"],dtype=float)
df4=pd.concat([df3,df3_dummies],axis=1)
df4.drop(columns="Quench_Proc",inplace=True)

Mask= (df4['Anneal_Temp'].notnull() & df4['Anneal_Time'].isnull())
df4mask=df4[~Mask]
df4mask.reset_index()

Mask2=(df4mask['Homog_Temp'].notnull() & df4mask['Homog_Time'].isnull())
df4mask2=df4mask[~Mask2].reset_index()
df4mask2["Homog_Temp"].fillna(0,inplace=True)
df4mask2["Homog_Time"].fillna(0,inplace=True)
df4mask2["Anneal_Temp"].fillna(0,inplace=True)
df4mask2["Anneal_Time"].fillna(0,inplace=True)
df4mask2.info()

df5=df4mask2.dropna()
df5.reset_index(inplace=True)

df6=df5.drop(columns=["level_0","index"])
df6.info()

HEA_data = df6
HEA_data

HEA_data.shape

HEA_data.head(10)

HEA_data.tail(10)

HEA_data.columns

HEA_data.dtypes

print(HEA_data.to_string())

HEA_data.shape

HEA_data_num_features = HEA_data.select_dtypes(exclude = "object")
HEA_data_num_features.head(10)
HEA_data.describe()


HEA_data_cat_features = new_HEA_data.select_dtypes(include = "object")
HEA_data_cat_features.head(10)
HEA_data_cat_features.shape


###LABEL ENCODING: CONVERT FROM CATEGORICAL VARIABLE TO NUMERICAL VARIABLE
label_enc = LabelEncoder()

HEA_data_cat_features["Sythesis_Route"] = label_enc.fit_transform(HEA_data_cat_features["Sythesis_Route"])

HEA_data_cat_features.tail(20)



# CARRY OUT ONE-HOT ENCODING ON CATEGORICAL ATTRIBUTES ON X
"""Sklearn OneHotEncoder()"""
categorical_columns = ['Sythesis_Route', 'Multiphase', 'Microstructure']

# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')

# Fit and transform the categorical columns
encoded_cols = encoder.fit_transform(HEA_data[categorical_columns])

# Get the new column names
new_column_names = encoder.get_feature_names_out(categorical_columns)

# Create a new DataFrame with the encoded variables
encoded_df = pd.DataFrame(encoded_cols, columns=new_column_names, index=HEA_data.index)

# Concatenate the encoded DataFrame with the original DataFrame, dropping the original categorical columns
final_HEA_data = pd.concat([HEA_data.drop(columns=categorical_columns), encoded_df], axis=1)
final_HEA_data.tail(20)

"""Pandas get_dummies()"""
HEA_data_cat_dummies = pd.get_dummies(HEA_data[categorical_columns], drop_first=True)
HEA_data_cat_dummies =  pd.concat([HEA_data, HEA_data_cat_dummies], axis = 1)

# CARRY OUT LABEL ENCODING ON y
# Change y(categorical variable) to numerical encoded variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(final_HEA_data['Phases'])
y_df = pd.DataFrame(y_encoded, columns=["Encoded_Phases"])
# Concatenate the encoded column with HEA_data
final_HEA_data = pd.concat([final_HEA_data, y_df], axis=1)

# If you want to replace the original "Phases" column with the encoded version:
if "Phases" in final_HEA_data.columns:
    final_HEA_data = final_HEA_data.drop(columns=["Phases"])
    final_HEA_data = final_HEA_data.rename(columns={"Encoded_Phases": "Phases"})

final_HEA_data.head(15)
'''Labels for Each Phase structure => FCC = 0, BCC = 1, FCC + BCC = 2, IM = 3'''

# Standardize data

# # List of numerical columns
# numerical_columns = ['Density_calc', 'Al', 'Co', 'Cr', 'Fe', 'Ni', 'Cu', 'Mn', 'Ti', 'V',
#                      'Nb', 'Mo', 'Zr', 'Hf', 'Ta', 'W', 'C', 'Mg', 'Zn', 'Si', 'Re', 'N',
#                      'Sc', 'Li', 'Sn', 'Be', 'VEC', 'Atom.Size.Diff', 'Elect.Diff', 'dHmix',
#                      'dSmix', 'dGmix', 'Tm', 'n.Para', 'Homog_Temp', 'Homog_Time',
#                      'Anneal_Temp', 'Anneal_Time']

# # Create a copy of the original DataFrame
# X_standardized = X.copy()

# # Initialize the StandardScaler
# standard_scaler = StandardScaler()
# # Fit and transform the numerical columns
# X_standardized[numerical_columns] = standard_scaler.fit_transform(X[numerical_columns])
# X_standardized.head()
# X_standardized.describe()

# Normalize data

# numerical_columns = ['Density_calc', 'Al', 'Co', 'Cr', 'Fe', 'Ni', 'Cu', 'Mn', 'Ti', 'V',
#                      'Nb', 'Mo', 'Zr', 'Hf', 'Ta', 'W', 'C', 'Mg', 'Zn', 'Si', 'Re', 'N',
#                      'Sc', 'Li', 'Sn', 'Be', 'VEC', 'Atom.Size.Diff', 'Elect.Diff', 'dHmix',
#                      'dSmix', 'dGmix', 'Tm', 'n.Para', 'Homog_Temp', 'Homog_Time',
#                      'Anneal_Temp', 'Anneal_Time']
# X_norm = X.copy()
# scaleMinMax = MinMaxScaler(feature_range = (0, 1))
# X_norm[numerical_columns] = scaleMinMax.fit_transform(X_norm[numerical_columns])
# # X_normalized = pd.DataFrame(X_normalized, columns = numerical_columns)
# X_norm.head()
# X_norm.describe()

# numerical_columns = ['Density_calc', 'Al', 'Co', 'Cr', 'Fe', 'Ni', 'Cu', 'Mn', 'Ti', 'V',
#                      'Nb', 'Mo', 'Zr', 'Hf', 'Ta', 'W', 'C', 'Mg', 'Zn', 'Si', 'Re', 'N',
#                      'Sc', 'Li', 'Sn', 'Be', 'VEC', 'Atom.Size.Diff', 'Elect.Diff', 'dHmix',
#                      'dSmix', 'dGmix', 'Tm', 'n.Para', 'Homog_Temp', 'Homog_Time',
#                      'Anneal_Temp', 'Anneal_Time']
# X_robust = X.copy()
# robust_scaling = RobustScaler()
# X_robust[numerical_columns] = robust_scaling.fit_transform(X_robust[numerical_columns])
# # X_normalized = pd.DataFrame(X_normalized, columns = numerical_columns)
# X_robust.head()
# X_robust.describe()

# TRAINING and TESTING DATASET SPLIT
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded_var, test_size = 0.2, train_size = 0.8, random_state = 42)

# Train-Test-Validation Split
# X_main, X_test, y_main, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# X_train, X_valid, y_train, y_valid = train_test_split(X_main, y_main, test_size = 0.2, stratify = y_main)

# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# BUILD AND TRAIN LOGISTIC REGRESSION MODEL WITH TRAIN DATA SET, THEN TEST MODEL PERFORMANCE
# log_reg = LogisticRegression(multi_class='multinomial', class_weight='balanced')
# log_model = log_reg.fit(X_train, y_train)  # Fits the model
# Make predictions
# y_pred = log_model.predict(X_test)
# y_pred_proba = log_model.predict_proba(X_test)
# Convert encoded predictions back to original labels
# y_pred_original = label_encoder.inverse_transform(y_pred)
# # Calculate and print accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)

# BUILD RANDOM FOREST MODEL WITH TRAIN DATA SET AND DECISION TREES, THEN TEST MODEL PERFORMANCE
# rf_classifier = RandomForestClassifier()
# rf_model = rf_classifier.fit(X_train, y_train)    #-Fits the model
# y_pred2 = rf_model.predict(X_test)  #-To make predictions
# y_pred_original2 = label_encoder.inverse_transform(y_pred2)
# y_pred_proba2 = rf_model.predict_proba(X_test)
# accuracy2 = accuracy_score(y_test, y_pred2)
# print(accuracy2)

# HEA_data["Phases"].unique()

# log_model.classes_

# cm = confusion_matrix(y_test, y_pred2,labels=rf_model.classes_ )
# ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=rf_model.classes_).plot()
# print(classification_report(y_test, y_pred2))



# Define the number of folds
# n_folds = 5

# Perform cross-validation for Logistic Regression
# log_reg_scores = cross_val_score(log_reg, X_robust, y, cv=n_folds, scoring='accuracy')

# Perform cross-validation for Random Forest
# rf_scores = cross_val_score(rf_classifier, X_robust, y, cv=n_folds, scoring='accuracy')

# Print results
# print("Logistic Regression Cross-Validation Results:")
# print(f"Accuracy: {np.mean(log_reg_scores):.4f} (+/- {np.std(log_reg_scores) * 2:.4f})")

# print("\nRandom Forest Cross-Validation Results:")
# print(f"Accuracy: {np.mean(rf_scores):.4f} (+/- {np.std(rf_scores) * 2:.4f})")

# Compare models
# print("\nModel Comparison:")
# if np.mean(log_reg_scores) > np.mean(rf_scores):
#     print("Logistic Regression performed better.")
# elif np.mean(rf_scores) > np.mean(log_reg_scores):
#     print("Random Forest performed better.")
# else:
#     print("Both models performed equally.")

