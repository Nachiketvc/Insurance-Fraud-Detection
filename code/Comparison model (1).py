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


#gaussian 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Read the dataset
data = pd.read_csv(r"C:\Users\nachi\OneDrive\Desktop\SEM4\insurance_data (1).csv")

# Selecting only the specified columns
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

# Convert 'fraud_reported' column to numerical values
label_encoder_fraud = LabelEncoder()
data['fraud_reported'] = label_encoder_fraud.fit_transform(data['fraud_reported'])

# Splitting the data into features and target
X = data.drop('fraud_reported', axis=1)
y = data['fraud_reported']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[4]:


# Random forest Classifier 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Read the dataset
data = pd.read_csv(r"C:\Users\nachi\OneDrive\Desktop\SEM4\insurance_data (1).csv")

# Selecting only the specified columns
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

# Convert 'fraud_reported' column to numerical values
label_encoder_fraud = LabelEncoder()
data['fraud_reported'] = label_encoder_fraud.fit_transform(data['fraud_reported'])

# Splitting the data into features and target
X = data.drop('fraud_reported', axis=1)
y = data['fraud_reported']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[5]:


#naivyes bayes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Read the dataset
data = pd.read_csv(r"C:\Users\nachi\OneDrive\Desktop\SEM4\insurance_data (1).csv")

# Selecting only the specified columns
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

# Convert 'fraud_reported' column to numerical values
label_encoder_fraud = LabelEncoder()
data['fraud_reported'] = label_encoder_fraud.fit_transform(data['fraud_reported'])

# Splitting the data into features and target
X = data.drop('fraud_reported', axis=1)
y = data['fraud_reported']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[6]:


#Multilayer perceptron
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Read the dataset
data = pd.read_csv(r"C:\Users\nachi\OneDrive\Desktop\SEM4\insurance_data (1).csv")

# Selecting only the specified columns
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

# Convert 'fraud_reported' column to numerical values
label_encoder_fraud = LabelEncoder()
data['fraud_reported'] = label_encoder_fraud.fit_transform(data['fraud_reported'])

# Splitting the data into features and target
X = data.drop('fraud_reported', axis=1)
y = data['fraud_reported']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the MLP model
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam', random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[7]:


#SVM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Read the dataset
data = pd.read_csv(r"C:\Users\nachi\OneDrive\Desktop\SEM4\insurance_data (1).csv")

# Selecting only the specified columns
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

# Convert 'fraud_reported' column to numerical values
label_encoder_fraud = LabelEncoder()
data['fraud_reported'] = label_encoder_fraud.fit_transform(data['fraud_reported'])

# Splitting the data into features and target
X = data.drop('fraud_reported', axis=1)
y = data['fraud_reported']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the SVM model
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[8]:


#XGBOOST
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Read the dataset
data = pd.read_csv(r"C:\Users\nachi\OneDrive\Desktop\SEM4\insurance_data (1).csv")

# Selecting only the specified columns
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

# Convert 'fraud_reported' column to numerical values
label_encoder_fraud = LabelEncoder()
data['fraud_reported'] = label_encoder_fraud.fit_transform(data['fraud_reported'])

# Splitting the data into features and target
X = data.drop('fraud_reported', axis=1)
y = data['fraud_reported']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the XGBoost model
model = XGBClassifier()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[10]:


#Logistic Regression 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Read the dataset
data = pd.read_csv(r"C:\Users\nachi\OneDrive\Desktop\SEM4\insurance_data (1).csv")

# Selecting only the specified columns
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

# Convert 'fraud_reported' column to numerical values
label_encoder_fraud = LabelEncoder()
data['fraud_reported'] = label_encoder_fraud.fit_transform(data['fraud_reported'])

# Splitting the data into features and target
X = data.drop('fraud_reported', axis=1)
y = data['fraud_reported']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Read the dataset
data = pd.read_csv(r"C:\Users\nachi\OneDrive\Desktop\SEM4\insurance_data (1).csv")

# Selecting only the specified columns
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

# Convert 'fraud_reported' column to numerical values
label_encoder_fraud = LabelEncoder()
data['fraud_reported'] = label_encoder_fraud.fit_transform(data['fraud_reported'])

# Splitting the data into features and target
X = data.drop('fraud_reported', axis=1)
y = data['fraud_reported']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(kernel='rbf', random_state=42),
    "Naive Bayes": GaussianNB(),
    "Multilayer Perceptron": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

# Test accuracy for each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: Accuracy = {accuracy}")


# In[ ]:





# In[17]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Read the dataset
data = pd.read_csv(r"C:\Users\nachi\OneDrive\Desktop\SEM4\insurance_data (1).csv")

# Selecting only the specified columns
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

# Convert 'fraud_reported' column to numerical values
label_encoder_fraud = LabelEncoder()
data['fraud_reported'] = label_encoder_fraud.fit_transform(data['fraud_reported'])

# Splitting the data into features and target
X = data.drop('fraud_reported', axis=1)
y = data['fraud_reported']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(kernel='rbf', random_state=42),
    "Naive Bayes": GaussianNB(),
    "Multilayer Perceptron": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

# Test accuracy, precision, recall, and F1 score for each classifier
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

# Print results
print("Results:")
for name, metrics in results.items():
    print(f"{name}: Accuracy = {metrics['Accuracy']}, Precision = {metrics['Precision']}, Recall = {metrics['Recall']}, F1 Score = {metrics['F1 Score']}")

# Plotting
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
plt.figure(figsize=(10, 6))
for metric in metrics_names:
    plt.plot(list(results.keys()), [metrics[metric] for metrics in results.values()], marker='o', label=metric)
plt.xlabel('Classifiers')
plt.ylabel('Metrics')
plt.title('Comparison of Metrics for Different Classifiers')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[ ]:




