#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Edit Working directory to where data file is located
import os

os.getcwd()
os.chdir("C:/Users/i_sli/Desktop/Introduction to Data Mining/Project")


# In[4]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from IPython.display import display
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import statsmodels.api as sm
pd.options.display.max_columns = None ### shows all columns
error_bad_lines = False
df=pd.read_csv('NYCgov_Poverty_Measure_Data__2016_.csv')


# ### Exploratory Data Analysis

# #### Yearly Income in poverty and non poverty groups

# In[6]:


poverty=df[df.NYCgov_Pov_Stat==1]
not_poverty=df[df.NYCgov_Pov_Stat==2]
plt.scatter(poverty.SERIALNO,poverty.NYCgov_Income)
plt.ylabel('Income')
plt.xlabel('Family Serial Number')
plt.show()


# In[7]:


plt.scatter(not_poverty.SERIALNO,not_poverty.NYCgov_Income)
plt.ylabel('Income')
plt.xlabel('Family Serial Number')
plt.show()


# ### Population in Non Poverty vs Population in Poverty
# 

# In[8]:


indx=['poverty','non_poverty']
bar=[len(poverty),len(not_poverty)]

plt.bar(indx, bar, alpha=0.5,color='r') ## alpha gives opacity
plt.title('Poverty count vs Non_Poverty count')
plt.show()

print(f"The count of poverty is {len(poverty)}, The count of Non_poverty is {len(not_poverty)}")
print(f"The percentage of non_poverty is {round(len(poverty)/len(not_poverty)*100)}")


# ### Poverty and non poverty in different towns of New York

# In[10]:


split_pov=poverty.groupby('Boro').Boro.count()
index_pov=split_pov.index
values_pov=split_pov.values

town_label=[1,2,3,4,5]
town_name=['Bronx','Brooklyn','Manhattan','Queens','Staten Island']
town_dict=dict(zip(town_label,town_name))

print(town_dict)
print(split_pov)

split_npov=not_poverty.groupby('Boro').Boro.count()
index_npov=split_npov.index
values_npov=split_npov.values

print(split_npov)


# In[11]:


fig, ax= plt.subplots()
bar_width = 0.35
opacity = 0.6

town_label=[1,2,3,4,5]
town_name=['Bronx','Brooklyn','Manhattan','Queens','Staten Island']
town_dict=dict(zip(town_label,town_name))
    
rects1 = plt.bar(split_pov.index, split_pov.values, bar_width, 
alpha=opacity,
color='b',
label='Poverty')
 
rects2 = plt.bar(split_pov.index + bar_width,split_npov.values , bar_width,
alpha=opacity,
color='g',
label='Not Poverty')
plt.xlabel('town')
plt.ylabel('Counts')
plt.legend()
plt.show()
print(town_dict)


# In[ ]:




