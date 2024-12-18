#!/usr/bin/env python
# coding: utf-8

# # Predict Loan Eligibility for Dream Housing Finance company
# Dream Housing Finance company deals in all kinds of home loans. They have presence across all urban, semi urban and rural areas. The customer first applies for home loan and after that company validates the customer eligibility for loan.
# 
# Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have provided a dataset to identify the customers segments that are eligible for loan amount so that they can specifically target these customers. 

# ## Train file: CSVcontaining the customers for whom loan eligibility is known as 'Loan_Status'.
# ##  Test file: CSVcontaining the customer information for whom loan eligibility is to be predicted
# # Data Dictionary
# 
# * Loan_ID:	Unique Loan ID
# * Gender:	Male/ Female
# * Married:	Applicant married (Y/N)
# * Dependents	Number of dependents
# * Education	Applicant Education (Graduate/ Under Graduate)
# * Self_Employed	Self employed (Y/N)
# * ApplicantIncome	Applicant income
# * Coapplicant Income: Income of coapplicant
# * LoanAmount:	Loan amount in thousands
# * Loan_Amount_Term:	Term of loan in months
# * Credit_History:	Credit history meets guidelines
# * Property_Area:	Urban/ Semi Urban/ Rural
# * Loan_Status(Target): Loan approved (Y/N)

# #### Loan_Status(Target) Loan approved (Y/N) is missing in test file since it is the Target.
# # Submission file format: 
# * Variable	Description
# * Loan_ID	Unique Loan ID
# * Loan_Status	(Target) Loan approved (Y/N)

# # Import the libraries

# In[335]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set_theme(color_codes=True)


# # Load Data

# In[338]:


train = pd.read_csv('train_ctrUa4K.csv')
train


# In[340]:


test = pd.read_csv('test_lAUu6dG.csv')
test


# # Data Preprocessing 1

# In[343]:


train.drop(columns=['Loan_ID'],inplace = True)
train


# # Find total number of rows and columns

# In[346]:


print("No.of rows in train data:",train.shape[0])
print("No.of columns in train data:",train.shape[1])
print("No.of rows in test data:",test.shape[0])
print("No.of columns in test data:",test.shape[1])


# # Check for duplicate rows

# In[349]:


train.duplicated().sum()


# In[351]:


test.duplicated().sum()


# # Find data type of each column and memory usage

# In[354]:


train.info()


# In[356]:


test.info()


# In[358]:


train.isnull().sum().sort_values(ascending=False)


# In[360]:


train['Credit_History'].fillna(0)
train['Self_Employed'].fillna('No')
train['LoanAmount'].fillna(0)
train['Dependents'].fillna('other')
train['Loan_Amount_Term'].fillna(0)
train['Gender'].fillna('other')
train['Married'].fillna('other')

train.fillna({
    'Credit_History': 0,
    'Self_Employed': 'No',
    'LoanAmount': 0,
    'Dependents': 'other',
    'Loan_Amount_Term': 0,
    'Gender': 'other',
    'Married': 'other',
}, inplace=True)


# In[362]:


train.isnull().sum()


# In[364]:


test.isna().sum()


# In[366]:


test['Credit_History'].fillna(0)
test['Self_Employed'].fillna('No')
test['LoanAmount'].fillna(0)
test['Dependents'].fillna('other')
test['Loan_Amount_Term'].fillna(0)
test['Gender'].fillna('other')
test['Married'].fillna('other')

test.fillna({
    'Credit_History': 0,
    'Self_Employed': 'No',
    'LoanAmount': 0,
    'Dependents': 'other',
    'Loan_Amount_Term': 0,
    'Gender': 'other',
    'Married': 'other',
}, inplace=True)


# In[368]:


test.isna().sum()


# # Exploratory Data Analysis

# In[371]:


sns.countplot(data=train, x="Loan_Status",hue="Property_Area")
#people with SemiUrban has high acceptable chance of Loan Status


# In[372]:


sns.countplot(data=train, x="Loan_Status",hue="Credit_History")
#people with accepatable past credit history are most likely accepted to new loan


# In[373]:


sns.countplot(data=train, x="Loan_Status",hue="Loan_Amount_Term")
#people with 360 month loan term are most likely to be acceptable


# In[375]:


sns.barplot(data=train, x="Loan_Status",y="LoanAmount")


# In[376]:


sns.barplot(data=train, x="Loan_Status",y="CoapplicantIncome")
#pwople with high coapplicant income are most not accepted to new loan


# In[378]:


sns.barplot(data=train, x="Loan_Status",y="ApplicantIncome")


# In[379]:


sns.countplot(data=train, x="Loan_Status",hue="Self_Employed")


# In[380]:


sns.countplot(data=train, x="Loan_Status",hue="Education")


# In[381]:


sns.countplot(data=train, x="Loan_Status",hue="Dependents")


# In[383]:


sns.countplot(data=train, x="Loan_Status",hue="Married")
#people who are married are ore acceptable to new loan


# In[384]:


sns.countplot(data=train, x="Loan_Status",hue="Gender")


# # Data Preprocessing 2

# In[387]:


train


# In[388]:


train['Gender'].unique()


# In[389]:


train['Married'].unique()


# In[390]:


train['Dependents'].unique()


# In[391]:


train['Self_Employed'].unique()


# In[392]:


train['Education'].unique()


# In[394]:


train['Property_Area'].unique()


# In[397]:


train['Loan_Status'].unique()


# In[399]:


train['Loan_Amount_Term'].unique()


# In[400]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
train['Gender']= label_encoder.fit_transform(train['Gender'])
train['Gender'].unique()


# In[401]:


train['Married']= label_encoder.fit_transform(train['Married'])
train['Married'].unique()


# In[402]:


train['Dependents']= label_encoder.fit_transform(train['Dependents'])
train['Dependents'].unique()


# In[404]:


train['Education']= label_encoder.fit_transform(train['Education'])
train['Education'].unique()


# In[406]:


train['Self_Employed']= label_encoder.fit_transform(train['Self_Employed'])
train['Self_Employed'].unique()


# In[407]:


train['Property_Area']= label_encoder.fit_transform(train['Property_Area'])
train['Property_Area'].unique()


# In[408]:


train['Loan_Amount_Term']= label_encoder.fit_transform(train['Loan_Amount_Term'])
train['Loan_Amount_Term'].unique()


# In[410]:


# Apply label encoding to multiple columns
train[['Gender', 'Married', 'Dependents', 'Education', 
      'Self_Employed', 'Property_Area', 'Loan_Amount_Term']] = train[
          ['Gender', 'Married', 'Dependents', 'Education', 
           'Self_Employed', 'Property_Area', 'Loan_Amount_Term']
      ].apply(label_encoder.fit_transform)

# To check unique values for encoded columns
print(train[['Gender', 'Married', 'Dependents', 'Education', 
            'Self_Employed', 'Property_Area', 'Loan_Amount_Term']])


# In[427]:


# Apply label encoding to multiple columns
test[['Gender', 'Married', 'Dependents', 'Education', 
      'Self_Employed', 'Property_Area', 'Loan_Amount_Term']] = test[
          ['Gender', 'Married', 'Dependents', 'Education', 
           'Self_Employed', 'Property_Area', 'Loan_Amount_Term']
      ].apply(label_encoder.fit_transform)

# To check unique values for encoded columns
print(test[['Gender', 'Married', 'Dependents', 'Education', 
            'Self_Employed', 'Property_Area', 'Loan_Amount_Term']])


# ## Outlier Handling

# In[431]:


sns.boxplot(x=train["ApplicantIncome"])


# In[432]:


sns.boxplot(x=train["CoapplicantIncome"])


# In[433]:


sns.boxplot(x=train["LoanAmount"])


# In[434]:


train.dtypes

import scipy.stats as stats
z = np.abs(stats.zscore(train))
data_clean= train[(z<3).all(axis= 1)]
data_clean.shape
# # Balanced Class Data
sns.countplot(data=data_clean, x="Loan_Status")
data_clean['Loan_Status'].value_counts()from sklearn.utils import resample
#create two different dataframe of majority and minority class
df_majority = data_clean[(data_clean['Loan_Status']==1)]
df_minority = data_clean[(data_clean['Loan_Status']==0)]
#upsample minority class
df_minority_upsampled = resample(df_minority,    #sample with replacement
                                 replace = True,  # to match majority class
                                 n_samples = 398, # reproducible results
                                 random_state=0)
#combine majority class with unsampled minority class
df_upsampled = pd.concat([df_minority_upsampled, df_majority])
sns.countplot(data=df_upsampled, x="Loan_Status")
df_upsampled['Loan_Status'].value_counts()
# ## Data Correlation

# In[ ]:


sns.heatmap(train.corr(), fmt='.2g')


# # Standard scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler,MinMaxScaler


# In[ ]:


from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
std_scale = StandardScaler()

# Fit the scaler on the training data
std_scale.fit(train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']])

# Transform both the training and test data using the fitted scaler
train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']] = std_scale.transform(train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']])
test[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']] = std_scale.transform(test[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']])


# In[479]:


train


# In[481]:


test


# ## Machine Learning Model Building

# In[486]:


X = train.drop('Loan_Status',axis=1)
y = train['Loan_Status']


# ## Random Forest

# In[489]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0)


# In[491]:


rfc.fit(X, y)

# Make predictions on the test data
y_pred_rfc = rfc.predict(test.drop(['Loan_ID','Loan_Status'],axis=1))


# In[493]:


test['Loan_Status']=y_pred_rfc
test[['Loan_ID','Loan_Status']].to_csv('loan_Prediction_rfc.csv',index=False)


# In[495]:


import os
print(os.getcwd())


# In[ ]:




