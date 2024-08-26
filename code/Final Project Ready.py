#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv(r"c:\Users\nachi\OneDrive\Desktop\SEM4\insurance_data (1).csv")
data.head()


# In[3]:


data.replace('?', np.nan, inplace = True)


# In[4]:


data.describe()


# In[5]:


data.info()


# In[6]:


data.isna().sum()


# In[7]:


unique = data.nunique().to_frame()
unique.columns = ['Count']
unique.index.names = ['ColName']
unique = unique.reset_index()
sns.set(style='whitegrid',color_codes=True)
sns.barplot(x='ColName',y='Count',data=unique)
plt.xticks(rotation=90)
plt.show()


# In[8]:


unique.sort_values(by='Count',ascending=False)


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have your dataset loaded into a DataFrame called 'data'
# 'data' should contain all your features including the target variable (if any)
# You may need to adjust this based on how your data is structured

# Drop the target variable if it's included in your DataFrame
# Example: data.drop('target_variable_name', axis=1, inplace=True)

X_numeric = data.select_dtypes(include=['number'])

plt.figure(figsize=(20, 15))
plotnumber = 1

for col in X_numeric.columns:
    if plotnumber <= 24:
        ax = plt.subplot(5, 5, plotnumber)
        sns.boxplot(X_numeric[col])
        plt.xlabel(col, fontsize=15)
    
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[10]:


# Assuming 'data' is your DataFrame
numeric_data = data.select_dtypes(include=[np.number])
corr = numeric_data.corr()

plt.figure(figsize=(18, 12))
sns.heatmap(data=corr, annot=True, fmt='.2g', linewidth=1)
plt.show()


# In[11]:


fraud_counts = data['fraud_reported'].value_counts()

# Plotting
plt.figure(figsize=(8, 6))
fraud_counts.plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Distribution of Fraudulent vs. Non-Fraudulent Cases')
plt.xlabel('Fraudulent')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
plt.show()


# In[12]:


severity_index=data.groupby(['incident_severity']).size().index
severity_values=data.groupby(['incident_severity']).size().values
colors=sns.color_palette('pastel')
plt.pie(x=severity_values,labels=severity_index,colors=colors,autopct='%.1f%%')
plt.title('Damage visualization')
plt.show()


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Selecting features based on correlation analysis
selected_columns = ['months_as_customer', 'policy_deductable', 'umbrella_limit', 'capital-gains',
                    'capital-loss', 'incident_hour_of_the_day', 'number_of_vehicles_involved',
                    'bodily_injuries', 'witnesses', 'vehicle_claim', 'collision_type', 'incident_severity',
                    'authorities_contacted', 'fraud_reported']
data = data[selected_columns]

# Encoding categorical variables
label_encoders = {}
categorical_columns = ['collision_type', 'incident_severity', 'authorities_contacted']
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Convert 'fraud_reported' column to numeric values
label_encoder_fraud = LabelEncoder()
data['fraud_reported'] = label_encoder_fraud.fit_transform(data['fraud_reported'])

# Splitting the data into features and target
X = data.drop('fraud_reported', axis=1)
y = data['fraud_reported']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
print("\nUpdated Classification Report:")
print(classification_report(y_test, y_pred))


# In[14]:


import seaborn as sns

# Select the specified columns along with the target variable
selected_columns_with_target = selected_columns

# Create a correlation matrix
corr_matrix = data[selected_columns_with_target].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[15]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Selecting features based on correlation analysis
selected_columns =  ['months_as_customer', 'policy_deductable', 'umbrella_limit', 'capital-gains',
                    'capital-loss', 'incident_hour_of_the_day', 'number_of_vehicles_involved',
                    'bodily_injuries', 'witnesses', 'vehicle_claim', 'collision_type', 'incident_severity',
                    'authorities_contacted', 'fraud_reported']

data = data[selected_columns]

# Encoding categorical variables
label_encoders = {}
categorical_columns = ['collision_type', 'incident_severity', 'authorities_contacted']
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Convert 'fraud_reported' column to numeric values
label_encoder_fraud = LabelEncoder()
data['fraud_reported'] = label_encoder_fraud.fit_transform(data['fraud_reported'])

# Splitting the data into features and target
X = data.drop('fraud_reported', axis=1)
y = data['fraud_reported']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a parameter grid for GridSearchCV
param_grid = {
    'learning_rate': [0.05, 0.1, 0.15],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Initialize XGBoost classifier
xgb = XGBClassifier()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Train XGBoost model with best parameters
best_xgb = XGBClassifier(**best_params)
best_xgb.fit(X_train, y_train)

# Make predictions
y_pred = best_xgb.predict(X_test)

# Evaluating the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[16]:


from xgboost import XGBClassifier
import pickle

# Assume you have your training data X_train, y_train ready

# Train XGBoost model
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

# Save the trained model to a file using pickle
with open('xgb_modeltestfinal99.pkl', 'wb') as f:
    pickle.dump(xgb, f)


# In[17]:


from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve

# Calculating ROC curve and AUC score
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)

# Additional metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nUpdated Classification Report:")
print(classification_report(y_test, y_pred))

# Printing additional metrics
print("\nAdditional Metrics:")
print(f"AUC Score: {auc_score}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")

# Plotting ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='orange', label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




