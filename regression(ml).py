#!/usr/bin/env python
# coding: utf-8

# # IMPORTING THE DEPENDENCIES

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # IMPORTING FUNCTIONS TO GET MODELS

# In[71]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


# # DATA COLLECTION and ANALYSIS

# In[8]:


#loading the dataset from csv file to a pandas dataframe
bm_data=pd.read_csv('big_marketTrain.csv')


# In[9]:


#to get first five rows of dataset
bm_data.head()


# In[10]:


# NUMBER OF DATA_POINTS(rows) AND NUMBER OF FEATURES(columns)
bm_data.shape


# In[11]:


# getting information about dataset
bm_data.info()


# # Categorical Features: string/object type
# # Item_Identifier,Item_Fat_Content,Item_Type,Outlet_Identifier,Outlet_Size,Outlet_Location_Type,Outlet_Type

# In[12]:


#Checking if table has missing values
bm_data.isnull().sum(axis=0)
#from this we conclude item_weight and outlet_size has missing values 1463 and 2410 respectively


# # Handling Missing Values
# # Mean --> average
# # Mode --> more repeated value

# In[13]:


#mean value of item_weight column
bm_data['Item_Weight'].mean()


# In[14]:


#filling the missing value in item_weight column with mean value
bm_data['Item_Weight'].fillna(bm_data['Item_Weight'].mean(),inplace=True)
#we use inplace to do permanent changes in orignal dataset


# In[15]:


#to check if missing values are present now also
bm_data.isnull().sum(axis=0)


# In[16]:


# replacing missing values using mode
mode_of_outletsize = bm_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
#create a table for outlet_type and size to fill missing values


# In[17]:


print(mode_of_outletsize)


# In[18]:


miss_values = bm_data['Outlet_Size'].isnull() 


# In[19]:


#array of true and false where false means not null and true means null 
print(miss_values)


# In[20]:


#locating values
bm_data.loc[miss_values, 'Outlet_Size'] = bm_data.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_outletsize[x])


# In[21]:


#checking for null values
bm_data.isnull().sum(axis=0)


# # DATA ANALYSIS

# In[22]:


#TO GET THE STATISTICAL INFORMATION ABOUT THE DATA
bm_data.describe()


# In[ ]:


#Numerical features


# In[23]:


#sns is used for making our plot visualisation better
sns.set()


# In[24]:


# Item_Weight distribution(meaning all information in graphical format)
plt.figure(figsize=(6,6))
sns.distplot(bm_data['Item_Weight'])
plt.show()


# In[25]:


# Item Visibility distribution
plt.figure(figsize=(6,6))
sns.distplot(bm_data['Item_Visibility'])
plt.show()


# In[26]:


# Item MRP distribution
plt.figure(figsize=(6,6))
sns.distplot(bm_data['Item_MRP'])
plt.show()


# In[27]:


# Item_Outlet_Sales distribution
plt.figure(figsize=(6,6))
sns.distplot(bm_data['Item_Outlet_Sales'])
plt.show()


# In[28]:


# Outlet_Establishment_Year column
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=bm_data)
plt.show()
#number of products sold in years


# In[29]:


# Item_Fat_Content column
plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content', data=bm_data)
plt.show()
#by this we can see data is not clean as 3 times low fat is coming..


# In[30]:


# Item_Type column
plt.figure(figsize=(30,6))
sns.countplot(x='Item_Type', data=bm_data)
plt.show()


# In[31]:


# Outlet_Size column
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Size', data=bm_data)
plt.show()


# # DATA PREPROCESSING

# In[32]:


#cleaning of data
bm_data['Item_Fat_Content'].value_counts()


# In[33]:


bm_data.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)


# In[34]:


#to check if working
bm_data['Item_Fat_Content'].value_counts()


# In[35]:


# to convert all categorical data into numerical data (using label encoding)
encoder = LabelEncoder()


# In[36]:


bm_data['Item_Identifier'] = encoder.fit_transform(bm_data['Item_Identifier'])

bm_data['Item_Fat_Content'] = encoder.fit_transform(bm_data['Item_Fat_Content'])

bm_data['Item_Type'] = encoder.fit_transform(bm_data['Item_Type'])

bm_data['Outlet_Identifier'] = encoder.fit_transform(bm_data['Outlet_Identifier'])

bm_data['Outlet_Size'] = encoder.fit_transform(bm_data['Outlet_Size'])

bm_data['Outlet_Location_Type'] = encoder.fit_transform(bm_data['Outlet_Location_Type'])

bm_data['Outlet_Type'] = encoder.fit_transform(bm_data['Outlet_Type'])


# In[37]:


bm_data.head()


# # SPLITTING FEATURES(all other columns) AND TARGET(outlet sale)

# In[38]:


X =bm_data.drop(columns='Item_Outlet_Sales', axis=1)#removing column
Y = bm_data['Item_Outlet_Sales']


# In[39]:


print(X)


# In[40]:


print(Y)


# # Splitting the Data into Testing and Training data
# 

# In[41]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[42]:


print(X.shape, X_train.shape, X_test.shape)


# # MACHINE LEARNING MODEL TRAINING

# # XGBregressor model

# In[43]:


regressor = XGBRegressor()


# In[44]:


regressor.fit(X_train, Y_train)#training our data(finding pattern between x and y)


# In[66]:


# prediction on training data
training_data_prediction = regressor.predict(X_train)
# R squared Value-in metrics library
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R Squared value = ', r2_train)


# In[67]:


# prediction on test data
test_data_prediction = regressor.predict(X_test)
# R squared Value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R Squared value = ', r2_test)


# In[101]:


regressor.score(X_train, Y_train)


# # Linear regression model

# In[68]:


model2=LinearRegression()
model2.fit(X_train,Y_train)


# In[69]:


# prediction on training data
training_data_prediction = model2.predict(X_train)
# R squared Value-in metrics library
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R Squared value = ', r2_train)


# In[84]:


# prediction on test data
test_data_prediction = model2.predict(X_test)
# R squared Value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R Squared value = ', r2_test)


# In[102]:


model2.score(X_train, Y_train)


# # LASSO MODEL

# In[74]:


model3=Lasso()
model3.fit(X_train,Y_train)
# prediction on training data
training_data_prediction = model3.predict(X_train)
from sklearn.metrics import mean_squared_error,r2_score
# R squared Value-in metrics library
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R Squared value = ', r2_train)


# In[80]:


# prediction on test data
test_data_prediction = model3.predict(X_test)
# R squared Value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R Squared value = ', r2_test)


# In[103]:


model3.score(X_train, Y_train)


# # RIDGE MODEL

# In[81]:


from sklearn.linear_model import Ridge
model4=Ridge()
model4.fit(X_train,Y_train)


# In[82]:


# prediction on training data
training_data_prediction = model4.predict(X_train)
from sklearn.metrics import mean_squared_error,r2_score
# R squared Value-in metrics library
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R Squared value = ', r2_train)


# In[83]:


# prediction on test data
test_data_prediction = model4.predict(X_test)
# R squared Value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R Squared value = ', r2_test)


# # ELASTICNET MODEL

# In[86]:


from sklearn.linear_model import ElasticNet
model5=ElasticNet()
model5.fit(X_train,Y_train)


# In[88]:


# prediction on training data
training_data_prediction = model5.predict(X_train)
from sklearn.metrics import mean_squared_error,r2_score
# R squared Value-in metrics library
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R Squared value = ', r2_train)


# In[87]:


# prediction on test data
test_data_prediction = model5.predict(X_test)
# R squared Value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R Squared value = ', r2_test)


# # SVM MODEL

# In[89]:


from sklearn.svm import SVR
model6=SVR()
model6.fit(X_train,Y_train)


# In[90]:


# prediction on training data
training_data_prediction = model6.predict(X_train)
from sklearn.metrics import mean_squared_error,r2_score
# R squared Value-in metrics library
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R Squared value = ', r2_train)


# In[91]:


# prediction on test data
test_data_prediction = model6.predict(X_test)
# R squared Value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R Squared value = ', r2_test)


# # RANDOMFOREST MODEL

# In[93]:


from sklearn.ensemble import RandomForestRegressor
model7=RandomForestRegressor()
model7.fit(X_train,Y_train)


# In[94]:


# prediction on training data
training_data_prediction = model7.predict(X_train)
from sklearn.metrics import mean_squared_error,r2_score
# R squared Value-in metrics library
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R Squared value = ', r2_train)


# In[95]:


# prediction on test data
test_data_prediction = model7.predict(X_test)
# R squared Value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R Squared value = ', r2_test)


# # DECISION TREE MODEL

# In[96]:


from sklearn.tree import DecisionTreeRegressor
model8=DecisionTreeRegressor()
model8.fit(X_train,Y_train)


# In[97]:


# prediction on training data
training_data_prediction = model8.predict(X_train)
from sklearn.metrics import mean_squared_error,r2_score
# R squared Value-in metrics library
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R Squared value = ', r2_train)


# In[98]:


# prediction on test data
test_data_prediction = model8.predict(X_test)
# R squared Value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R Squared value = ', r2_test)


# In[ ]:




