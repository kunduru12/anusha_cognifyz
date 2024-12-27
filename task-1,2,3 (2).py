#!/usr/bin/env python
# coding: utf-8

# # Task_1 

# Task: Table Booking and Online Delivery

# Determine the percentage of restaurants that
# offer table booking and online delivery.
# Compare the average ratings of restaurants
# with table booking and those without.
# Analyze the availability of online delivery
# among restaurants with different price ranges.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('Dataset .csv')
df.head(5)


# # Determine the percentage of restaurants that offer table booking and online delivery

# In[3]:


# the percentage of restaurants offering table booking
table_booking_percentage_yes = (df['Has Table booking'] == 'Yes').mean() * 100
print(table_booking_percentage_yes)
table_booking_percentage_no = (df['Has Table booking'] == 'Yes').mean() * 100
print(table_booking_percentage_no)


# In[4]:


# the percentage of restaurants offering online delivery
online_delivery_percentage_yes = (df['Has Online delivery'] == 'Yes').mean() * 100
print(online_delivery_percentage_yes)
online_delivery_percentage_no = (df['Has Online delivery'] == 'No').mean() * 100
print(online_delivery_percentage_no)


# # Compare the average ratings of restaurants with table booking and those without

# In[5]:


# Group data by 'Table_Booking' and calculate Aggregate rating
df1= df.groupby('Has Table booking')['Aggregate rating'].mean()
df1


# In[6]:


df1.plot(kind='bar',color='green')
plt.xlabel('Table Booking')
plt.ylabel('Average Rating')
plt.title('Average Rating by Table Booking')
plt.show()


# # Analyze the availability of online delivery among restaurants with different price ranges.

# In[7]:


# Group data by price range and calculate the percentage of restaurants with online delivery
price_ranges= df.groupby('Price range')['Has Online delivery'].value_counts(normalize=True).unstack()*100
price_ranges


# In[8]:


# Visualize the results
plt.figure(figsize=(10, 6))
price_ranges.plot(kind='bar')
plt.xlabel('Price Range')
plt.ylabel('Percentage of Restaurants with Online Delivery')
plt.title('Online Delivery Availability by Price Range')
plt.legend()
plt.show()


# In[9]:


#calculate the percentage of restaurants with online delivery
online_delivery_yes = df[df['Has Online delivery'] == 'Yes']
online_delivery_counts = online_delivery_yes.groupby('Price range').size()
online_delivery_counts


# In[10]:


# Visualize the results
plt.figure(figsize=(10, 6))
online_delivery_counts.plot(kind='bar')
plt.xlabel('Price Range')
plt.ylabel('number of Restaurants')
plt.title('Online Delivery Availability by Price Range')
plt.show()


# # Task-2

# #Task: Price Range Analysis 

# Determine the most common price range
# among all the restaurants.
# Calculate the average rating for each price
# range.
# Identify the color that represents the highest
# average rating among different price ranges.

# # Determine the most common price range among all the restaurants.

# In[11]:


df["Price range"].value_counts()


# # Calculate the average rating for each price range.

# In[12]:


average_rating_by_price_range = df.groupby('Price range')['Aggregate rating'].mean()
print(average_rating_by_price_range)


# # Identify the color that represents the highest average rating among different price ranges.

# In[13]:



highest_rating_price_range = average_rating_by_price_range.idxmax()

# Create a color palette
colors = ['blue', 'green', 'red','orange']

# Create a bar chart
plt.bar(average_rating_by_price_range.index,average_rating_by_price_range, color=colors)
plt.xlabel('Price Range')
plt.ylabel('Average Rating')
plt.title('Average Rating by Price Range')
plt.show()


# In[14]:


# Print the color associated with the highest rating
print(f"The color representing the highest average rating is:{colors[average_rating_by_price_range.index.get_loc(highest_rating_price_range)]}")


# # Task-3

# Task: Feature Engineering

# Extract additional features from the existing
# columns, such as the length of the restaurant
# name or address.
# Create new features like "Has Table Booking"
# or "Has Online Delivery" by encoding
# categorical variables.

# # Extract additional features from the existing columns, such as the length of the restaurant name or address.

# In[15]:


Name_Length = df['Restaurant Name'].str.len()
Name_Length


# In[16]:


Address_Length = df['Address'].str.len()
Address_Length


# In[17]:


Name_Count = df['Restaurant Name'].str.split().str.len()
Name_Count


# In[18]:


Address_Count= df['Address'].str.split().str.len()
Address_Count


# #  Create new features like "Has Table Booking" or "Has Online Delivery" by encoding categorical variables.

# In[19]:


df['Has Table Booking1'] = df['Has Table booking'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Has Online Delivery1'] = df['Has Online delivery'].apply(lambda x: 1 if x == 'Yes' else 0)
print(df)


# In[ ]:




