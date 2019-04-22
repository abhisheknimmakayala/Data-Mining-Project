#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from IPython.display import display
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.max_columns = None ### shows all columns
df=pd.read_csv('NYCgov_Poverty_Measure_Data__2016_.csv')


# In[18]:


plt.figure(figsize=(12,10))
sns.heatmap(df2.corr(), cbar=True)
plt.show()


# # Decision Tree Raw Data
# 

# In[22]:



rawdf=pd.read_csv('NYCgov_Poverty_Measure_Data__2016_.csv')
rawdf.head()


# In[23]:


rawdf.loc[:,'NYCgov_Pov_Stat']
rawdf.columns.get_loc("NYCgov_Pov_Stat")


# In[65]:


cols = rawdf.columns.tolist()
cols.insert(79, cols.pop(cols.index('NYCgov_Pov_Stat')))
cols
rawdf = rawdf.reindex(columns= cols)
rawdf = rawdf.dropna()
rawdf = rawdf.drop(['PreTaxIncome_PU','NYCgov_IncomeTax','NYCgov_FICAtax','NYCgov_SNAP', 'NYCgov_WIC', 'NYCgov_SchoolBreakfast', 'NYCgov_SchoolLunch' , 'NYCgov_HEAP' , 'NYCgov_Housing' ,
                    'NYCgov_Commuting', 'NYCgov_Childcare' ,'NYCgov_MOOP'],axis=1)


# In[83]:


rawdf.columns


# In[106]:


#rawdf = rawdf.drop('NYCgov_Threshold',axis=1)
#rawdf = rawdf.drop('Off_Threshold',axis=1)
rawdf = rawdf.drop('Off_Pov_Stat',axis=1)


# In[107]:


train, test = train_test_split(rawdf, test_size=0.2)
testY = test.iloc[:,-1]
testX = test.iloc[:,:-1]


# In[108]:



from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
from sklearn import tree


# In[109]:


clf = DecisionTreeClassifier(max_depth=4)
clf.fit(train.iloc[:,:-1],train.iloc[:,-1])


# In[110]:


print("Accuracy on training set:",clf.score(train.iloc[:,:-1],train.iloc[:,-1]))
print("Accuracy on testing set:", clf.score(testX,testY))


# In[111]:


dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,feature_names=list(rawdf.columns[:-1]))
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# In[ ]:





# # Decision Tree with Missing value replacement
# 

# In[144]:


Newdf = df


# In[166]:


len(df.columns)-len(Newdf.columns)


# In[145]:


Newcols = Newdf.columns.tolist()
Newcols.insert(79, Newcols.pop(Newcols.index('NYCgov_Pov_Stat')))
Newdf = Newdf.reindex(columns= Newcols)


# In[146]:


Newdf.head()
Newdf.columns


# In[147]:


Newdf = Newdf.drop(['PreTaxIncome_PU','NYCgov_IncomeTax','NYCgov_FICAtax','NYCgov_SNAP', 'NYCgov_WIC', 'NYCgov_SchoolBreakfast', 'NYCgov_SchoolLunch' , 'NYCgov_HEAP' , 'NYCgov_Housing' ,
                    'NYCgov_Commuting', 'NYCgov_Childcare' ,'NYCgov_MOOP','NYCgov_PovGap','NYCgov_PovGapIndex',
                   'Off_Pov_Stat','NYCgov_Threshold','Off_Threshold'],axis=1)
Newdf = Newdf.drop('NYCgov_Income',axis=1)


# In[148]:


train, test = train_test_split(Newdf, test_size=0.2)
testY = test.iloc[:,-1]
testX = test.iloc[:,:-1]


# In[149]:


clf = DecisionTreeClassifier(max_depth=3)
clf.fit(train.iloc[:,:-1],train.iloc[:,-1])


# In[150]:


print("Accuracy on training set:",clf.score(train.iloc[:,:-1],train.iloc[:,-1]))
print("Accuracy on testing set:", clf.score(testX,testY))


# In[169]:


dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,feature_names=list(Newdf.columns[:-1]))
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# In[142]:


print(clf.feature_importances_)


# In[159]:


def plot_feature_important(model):
    len_features = len(Newdf.columns[:-1])
    plt.barh(np.arange(len_features),model.feature_importances_,align = 'center')
    plt.yticks(np.arange(len_features),Newdf.columns)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.ylim(-1,len_features)
    


# In[161]:



plt.figure(figsize=(12,10))
plot_feature_important(clf)


# In[162]:


from sklearn.ensemble import RandomForestClassifier


# In[163]:


forest = RandomForestClassifier(n_estimators=100,max_features=4)
forest.fit(train.iloc[:,:-1],train.iloc[:,-1])


# In[164]:


print("Accuracy on training set:",forest.score(train.iloc[:,:-1],train.iloc[:,-1]))
print("Accuracy on testing set:", forest.score(testX,testY))


# In[165]:



plt.figure(figsize=(12,10))
plot_feature_important(forest)


# In[170]:


single = forest.estimators_[45]
tree.export_graphviz(single, out_file=dot_data,feature_names=list(Newdf.columns[:-1]))
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# In[ ]:




