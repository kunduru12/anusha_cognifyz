#!/usr/bin/env python
# coding: utf-8

# # Task-1

# # Task: Data Exploration and Preprocessing

# Explore the dataset and identify the number
# of rows and columns.
# Check for missing values in each column and
# handle them accordingly.
# Perform data type conversion if necessary.
# Analyze the distribution of the target variable
# ("Aggregate rating") and identify any class
# imbalances.

# In[1]:


#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('Dataset .csv') #loading dataset
df.head(10)


# In[3]:


df.shape  #the number of rows and columns


# In[4]:


df.columns #column names


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.isnull().any() #checking missing values in each column


# In[8]:


df.isnull().sum() #counting missing values in each column


# In[9]:


df1=df.fillna(method='pad')#  Use fillna with method='pad' for object columns based on previous one.
df1


# In[10]:


df1.isnull().sum()


# In[11]:


df1.duplicated().any()


# In[12]:


#Analyze the distribution of the target variable ("Aggregate rating") 
target="Aggregate rating"
df1[target].describe()


# In[13]:


# using data visualization:identify any class imbalances.
sns.histplot(df1[target],bins=10,kde=True,color='red')
plt.title('class imbalances.')
plt.xlabel('Aggregate rating')
plt.ylabel('frequency')
plt.show()


# In[14]:


sns.violinplot(df1[target])
plt.xlabel('Aggregate rating')
plt.show()


# # Task-2

# #Task: Descriptive Analysis 

# Calculate basic statistical measures (mean, median, standard deviation, etc.) for numerical columns. Explore the distribution of categorical variables like "Country Code,City and Cuisines". Identify the top cuisines and cities with the highest number of restaurants.

# In[15]:


#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


df=pd.read_csv('Dataset .csv')
df.head(5)


# In[17]:


#Calculate basic statistical measures (mean,median, standard deviation, etc.) for numerical columns.
df.describe()


# In[18]:


#Explore the distribution of categorical variables like "Country Code,City,and Cuisines.
sns.histplot(x='Country Code', data=df,color='red')
plt.title('distribution of country code by Restaurants')
plt.xlabel('Country Code')
plt.ylabel('restaurants')
plt.show()


# In[19]:


plt.figure(figsize=(10,8))
sns.histplot(x='City',data=df,color='red')
plt.title('distribution of City by Restaurants')
plt.xlabel('City')
plt.ylabel('restaurants')
plt.xticks(rotation=75)
plt.show()


# In[20]:


sns.histplot(x='Cuisines', data=df,color='green')
plt.title('distribution of Cuisines by Restaurants')
plt.xlabel('Cuisines')
plt.ylabel('restaurants')
plt.show()


# In[21]:


#Identify the top cuisines and cities with the highest number of restaurants.
highest_countries=df["Country Code"].value_counts().head()
highest_countries


# In[22]:


highest_cities=df["City"].value_counts().head(10)
highest_cities


# In[23]:


highest_Cuisines=df["Cuisines"].value_counts().head(10)
highest_Cuisines


# # Task-3

# Task: Geospatial Analysis

# Visualize the locations of restaurants on a map using latitude and longitude information. Analyze the distribution of restaurants across different cities or countries. Determine if there is any correlation between the restaurant's location and its rating.

# In[24]:


#importing required libraries
import folium


# In[25]:


df=pd.read_csv('Dataset .csv')
df.head(5)


# In[31]:


# Create a pandas dataframe
df1 = pd.DataFrame(df)

latitude = df1['Latitude'].mean()
longitude = df1['Longitude'].mean()

# Create a folium map
map = folium.Map(location=[latitude, longitude], zoom_start=12 ,width=1000,height=500)

# Add markers to the map for each restaurant
for i in range(0, len(df1)):
    folium.Marker(
        [df1.iloc[i]['Latitude'], df1.iloc[i]['Longitude']],
        popup=df1.iloc[i]['Restaurant Name'] +  (" + df1.iloc[i]['Cuisines'] + ")).add_to(map)

# Display the map
map


# In[27]:


# Analyze the distribution of restaurants across different cities or countries.
# Analyzing distribution by city
city_distribution = df1['City'].value_counts()

# Analyzing distribution by country
country_code_distribution = df1['Country Code'].value_counts()

# Display top results for both city and country distributions
city_distribution.head(), country_code_distribution.head()


# In[28]:


plt.figure(figsize=(8,5))
sns.countplot(x=df1['City'],order=df1.City.value_counts().head(5).index,color='green')
plt.title('Analyze the distribution of restaurants across different cities')
plt.xlabel('restaurants')
plt.ylabel('Cities')


# In[29]:


# Determine if there is any correlation between the restaurant's location and its rating.
plt.figure(figsize=(8,5))
df2=df1[['Latitude','Longitude','Aggregate rating']].corr()
sns.heatmap(df2,annot=True,cmap='coolwarm',fmt='.2f')
plt.title('correlation between the restaurants location and its rating')
plt.show()


# In[ ]:




