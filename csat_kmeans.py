#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


# In[2]:


feedback_df = pd.read_csv(r"C:\Users\pc\Desktop\kaggle\input\Customer_support_data.csv")
feedback_df


# In[3]:


feedback_df.info()


# In[26]:


feedback_df.describe()


# In[27]:


feedback_df.shape


# In[ ]:


feedback_df.columns


# In[8]:


csat = feedback_df[['Unique id']].groupby(feedback_df['CSAT Score']).count()
#csat.columns=['score','count']

csat


# # Counts of CSAT scores

# In[ ]:


fig, ax = plt.subplots()
colours = ['black', 'red', 'orange', 'yellow', 'green']
ax.bar(csat.index, csat['Unique id'], color=colours)
ax.set_title = 'Counts of CSAT scores'
ax.set_xlabel('CSAT')
ax.set_ylabel('Count')
plt.show()


# In[12]:


sns.scatterplot(data=csat,
            x="CSAT Score",
                y="Unique id",
            hue="CSAT Score")


# In[ ]:


# # Distribution of Item price, connected handling time and CSAT Score

# In[31]:


numerical_columns = ['Item_price', 'connected_handling_time', 'CSAT Score']

for col in numerical_columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(feedback_df[col], kde=True, bins=20)
    plt.title(f'Distribution of {col}')
    plt.show()


# In[30]:

numerical_columns = feedback_df.select_dtypes(include=[np.number])

plt.figure(figsize=(10, 8))
sns.heatmap(numerical_columns.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# In[ ]:


categorical_columns = ['category', 'Sub-category', 'Agent Shift', 'Tenure Bucket']

for col in categorical_columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=feedback_df, x=col)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()


# In[28]:


#  CSAT Score distribution by Cluster

# In[34]:

# Plotting CSAT Score by Cluster
if 'Cluster' in feedback_df.columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=feedback_df, x='Cluster', y='CSAT Score')
    plt.title('CSAT Score by Cluster')
    plt.show()
else:
    print("Cluster column not found. Please check the DataFrame.")
# Average CSAT score per cluster

# In[35]:

numerical_columns = ['Item_price', 'connected_handling_time', 'CSAT Score']
feedback_df = feedback_df.fillna(feedback_df.mean(numeric_only=True))
scaler = StandardScaler()
scaled_data = scaler.fit_transform(feedback_df[numerical_columns])


kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(scaled_data)


feedback_df['Cluster'] = cluster_labels
csat_avg = feedback_df.groupby('Cluster')['CSAT Score'].mean().reset_index()
csat_avg


# In[36]:


label_encoders = {}
categorical_columns = ['category', 'Sub-category', 'Agent Shift', 'Tenure Bucket']
for col in categorical_columns:
    le = LabelEncoder()
    feedback_df[col] = le.fit_transform(feedback_df[col].astype(str))  # Convert to string to handle NaNs as well
    label_encoders[col] = le


# In[37]:


feedback_df = feedback_df.fillna(feedback_df.mean(numeric_only=True))


# In[38]:


feedback_df.head()


# In[39]:


scaler = StandardScaler()
scaled_data = scaler.fit_transform(feedback_df.select_dtypes(include=[np.number]))


# In[40]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)


# In[41]:


plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:


for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(scaled_data)
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    print(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg}")


# In[ ]:


optimal_k = 3 
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(scaled_data)


# In[23]:



feedback_df['Cluster'] = cluster_labels


feedback_df.head()

import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


feedback_df = feedback_df.fillna(feedback_df.mean(numeric_only=True))
numerical_columns = ['Item_price', 'connected_handling_time', 'CSAT Score']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(feedback_df[numerical_columns])

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(scaled_data)


with open('kmeans_model.pkl', 'wb') as model_file:
    pickle.dump(kmeans, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
