#!/usr/bin/env python
# coding: utf-8

# # Task-1

# # Task: Predictive Modeling

# Build a regression model to predict the
# aggregate rating of a restaurant based on
# available features.
# Split the dataset into training and testing sets
# and evaluate the model's performance using
# appropriate metrics.
# Experiment with different algorithms (e.g.,
# linear regression, decision trees, random
# forest) and compare their performance.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df1=pd.read_csv('Dataset .csv')
df1.head(5)


# In[3]:


df1.info()


# In[4]:


df1.describe()


# In[5]:


df1.isna().any()


# In[6]:


df1.fillna(method='pad')


# In[7]:


df1.isnull().sum()


# In[8]:


df1.duplicated()


# # Build a regression model to predict the aggregate rating of a restaurant based on available features

# In[9]:


df1['Has Table Booking1'] = df1['Has Table booking'].apply(lambda x: 1 if x == 'Yes' else 0)
df1['Has Online Delivery1'] = df1['Has Online delivery'].apply(lambda x: 1 if x == 'Yes' else 0)
print(df1)


# In[10]:


x=df1[['Has Table Booking1','Has Online Delivery1','Price range','Votes']]
x


# In[11]:


y=df1[['Aggregate rating']]
y


# #Split the dataset into training and testing sets and evaluate the model's performance using appropriate metrics

# In[12]:


from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score


# In[13]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)


# In[14]:


x_train


# In[15]:


x_test


# In[16]:


y_train


# In[17]:


y_test


# In[18]:


model= linear_model.LinearRegression()
model.fit(x_train,y_train)


# In[19]:


y_pred=model.predict(x_test)
y_pred


# In[20]:


mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(mse)
print(r2)


# # Experiment with different algorithms (e.g., linear regression, decision trees, random forest) and compare their performance.

# In[21]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# In[22]:


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}


# In[23]:


results = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R^2': r2}
    
for model_name, metrics in results.items():
    print(f"{model_name}:\n\tMSE: {metrics['MSE']:.2f}\n\tR^2: {metrics['R^2']:.2f}\n")


# In[24]:


# Prepare data for plotting
metrics = ['MSE', 'R^2']
values = {metric: [results[model][metric] for model in results] for metric in metrics}


# In[25]:


for metric in metrics:
    plt.figure(figsize=(5, 5))
    plt.bar(results.keys(), values[metric], color='skyblue' if metric == 'MSE' else 'lightgreen')
    plt.title(f'Model Comparison ({metric})')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.xticks(rotation=15)
    plt.show()


# # Task-2: Customer Preference Analysis

# Analyze the relationship between the type of
# cuisine and the restaurant's rating.
# Identify the most popular cuisines among
# customers based on the number of votes.
# Determine if there are any specific cuisines
# that tend to receive higher ratings.
# 

# In[26]:


cuisines=df1['Cuisines']
cuisines
cuisines.value_counts().head(10)


# In[27]:


# Get top 10 cuisines by average rating
top_10_cuisines =cuisines.value_counts().head(10).index
top_10_cuisines


# In[28]:


cuisine_ratings=pd.DataFrame({'Cuisine':cuisines,'Rating':df1['Aggregate rating']})
cuisine_ratings


# In[29]:


# Filter the original DataFrame to include only top 10 cuisines
cuisine_ratings_top_10=cuisine_ratings[cuisine_ratings['Cuisine'].isin(top_10_cuisines)]
cuisine_ratings_top_10


# # Analyze the relationship between the type of cuisine and the restaurant's rating
# 
# 

# In[30]:


# Create a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(cuisine_ratings_top_10,x='Cuisine',y='Rating', palette='viridis')
plt.xlabel('cuisines')
plt.ylabel('Aggregate Rating')
plt.title('Boxplot of Aggregate Ratings by Cuisine')
plt.show()


# # Identify the most popular cuisines among customers based on the number of votes. 

# In[31]:


# Count the number of votes for each cuisine
cuisine_votes=pd.DataFrame({'Cuisine':cuisines,'Votes':df1['Votes']})
cuisine_votes


# In[32]:


# Get the top 10 most popular cuisines
cuisine_votes_sum = cuisine_votes.groupby('Cuisine')['Votes'].sum()
popular_cuisines=cuisine_votes_sum.sort_values(ascending=False)
popular_cuisines.head(10)


# In[33]:


plt.figure(figsize=(10, 6))
popular_cuisines.head(10).plot(kind='bar')
plt.xlabel('Cuisines')
plt.ylabel('Total Votes')
plt.title('Top 10 Most Popular Cuisines by Votes')
plt.xticks(rotation=45)
plt.show()


# # Determine if there are any specific cuisines that tend to receive higher ratings.

# In[34]:


cuisine_ratings=pd.DataFrame({'Cuisine':cuisines,'Rating':df1['Aggregate rating']})
cuisine_ratings


# In[35]:


average_rating_by_cuisine = cuisine_ratings.groupby('Cuisine')['Rating'].mean()


# In[36]:


sorted_cuisine_by_rating=average_rating_by_cuisine.sort_values(ascending=False)
sorted_cuisine_by_rating.head(10)


# In[37]:


#Create boxplot
plt.figure(figsize=(10, 6))
sorted_cuisine_by_rating.head(10).plot(kind='barh')
plt.xlabel('Aggregate rating')
plt.ylabel('Cuisines')
plt.title(' highest Average ratings Distribution for Top 10 Cuisines')
plt.show()


# # Task-3

# # Task: Data Visualization

# Create visualizations to represent the distribution
# of ratings using different charts (histogram, bar
# plot, etc.).
# Compare the average ratings of different cuisines
# or cities using appropriate visualizations.
# Visualize the relationship between various
# features and the target variable to gain insights.

# # Create visualizations to represent the distribution of ratings using different charts (histogram, barplot, etc.)

# In[38]:


# Plot 1: Histogram of Aggregate Ratings
plt.figure(figsize=(10, 6))
sns.histplot(df1['Aggregate rating'], bins=10, kde=True, color='skyblue', edgecolor='black')
plt.title('Histogram of Aggregate Ratings')
plt.xlabel('Aggregate Rating')
plt.ylabel('Frequency')
plt.show()


# In[39]:



# Plot 2: Bar Plot of Average Ratings
plt.figure(figsize=(10, 6))
rating_counts = df1['Aggregate rating'].value_counts().sort_index()
sns.barplot(x=rating_counts.index, y=rating_counts.values, palette='viridis')
plt.title('Bar Plot of Aggregate Ratings')
plt.xlabel('Aggregate Rating')
plt.ylabel('Count')
plt.grid(True)
plt.show()


# # Compare the average ratings of different cuisines or cities using appropriate visualizations.

# In[40]:


# Group by 'Cuisines' and 'City' and calculate average ratings
cuisine_ratings = df1.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False).head(10)
city_ratings = df1.groupby('City')['Aggregate rating'].mean().sort_values(ascending=False).head(10)

# Plot 1: Average Ratings by Top 10 Cuisines
plt.figure(figsize=(12, 6))
sns.barplot(x=cuisine_ratings.values, y=cuisine_ratings.index, palette='coolwarm')
plt.title('Average Ratings by Top 10 Cuisines')
plt.xlabel('Average Rating')
plt.ylabel('Cuisine')
plt.show()


# In[41]:


# Plot 2: Average Ratings by Top 10 Cities
plt.figure(figsize=(12, 6))
sns.barplot(x=city_ratings.values, y=city_ratings.index, palette='magma')
plt.title('Average Ratings by Top 10 Cities')
plt.xlabel('Average Rating')
plt.ylabel('City')
plt.show()


# # Visualize the relationship between various features and the target variable to gain insights.

# In[42]:


# 1. Scatter Plot: Votes vs. Aggregate Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(df1, x='Votes', y='Aggregate rating', color='blue')
plt.title('Votes vs. Aggregate Rating')
plt.xlabel('Votes')
plt.ylabel('Aggregate Rating')
plt.grid(alpha=0.7)
plt.show()


# In[43]:


# 2. Box Plot: Price Range vs. Aggregate Rating
plt.figure(figsize=(10, 6))
sns.boxplot(df1, x='Price range', y='Aggregate rating', palette='pastel')
plt.title('Price Range vs. Aggregate Rating')
plt.xlabel('Price Range')
plt.ylabel('Aggregate Rating')
plt.grid(alpha=0.7)
plt.show()


# In[44]:


# 3. Bar Plot: Average Aggregate Rating by City (Top 10 Cities)
city_avg_ratings = df1.groupby('City')['Aggregate rating'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=city_avg_ratings.values, y=city_avg_ratings.index, palette='coolwarm')
plt.title('Average Aggregate Rating by City (Top 10)')
plt.xlabel('Average Rating')
plt.ylabel('City')
plt.grid(axis='x', alpha=0.7)
plt.show()


# In[45]:


# 4. Bar Plot: Average Aggregate Rating by Cuisines (Top 10 Cuisines)
cuisine_avg_ratings = df1.groupby('Cuisines')['Aggregate rating'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=cuisine_avg_ratings.values, y=cuisine_avg_ratings.index, palette='viridis')
plt.title('Average Aggregate Rating by Cuisines (Top 10)')
plt.xlabel('Average Rating')
plt.ylabel('Cuisine')
plt.show()


# In[ ]:




