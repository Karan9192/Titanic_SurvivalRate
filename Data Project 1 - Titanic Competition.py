#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from pandas import Series, DataFrame


# In[3]:


titanic_df = pd.read_csv('train.csv')


# In[4]:


titanic_df.head()


# In[5]:


titanic_df.info()


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[10]:


sns.catplot(x='Sex',data=titanic_df, kind="count")


# In[14]:


sns.catplot('Sex',data=titanic_df,kind='count',hue='Pclass')


# In[44]:


def male_female_child(passenger):
    # Take the Age and Sex
    a,s = passenger
    # Compare the age, otherwise leave the sex
    if a < 16:
        return 'child'
    else:
        return s


# In[48]:


titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)


# In[49]:


titanic_df[0:10]


# In[36]:


sns.catplot('Pclass',data=titanic_df,kind='count',hue='person')


# In[38]:


titanic_df['Age'].hist(bins=70)


# In[51]:


titanic_df['person'].value_counts()


# In[53]:


fig = sns.FacetGrid(titanic_df, hue="Sex",aspect=4)

fig.map(sns.kdeplot,'Age',shade= True)


oldest = titanic_df['Age'].max()


fig.set(xlim=(0,oldest))


fig.add_legend()


# In[54]:


# Same thing for the 'person' column to include children:

fig = sns.FacetGrid(titanic_df, hue="person",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


# In[55]:


# For class, changing the hue argument
fig = sns.FacetGrid(titanic_df, hue="Pclass",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


# In[59]:


#Dropping NaN values in Cabin field
deck = titanic_df['Cabin'].dropna()
deck.head()


# In[62]:


# Grabbing first alphabet to denote level

# Set empty list
levels = []

# Loop to grab first letter
for x in deck:
    levels.append(x[0])    

# Reset DataFrame and use factor plot
cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.countplot('Cabin',data=cabin_df,palette='winter_d')


# In[64]:


# Redefining cabin_df to remove T cabin
cabin_df = cabin_df[cabin_df.Cabin != 'T']
#Replot
sns.countplot('Cabin',data=cabin_df,palette='summer')


# In[67]:


#Visualizing where the passengers came from
sns.countplot('Embarked',data=titanic_df,hue='Pclass')


# In[69]:


# define alone

titanic_df['Alone'] =  titanic_df.Parch + titanic_df.SibSp
titanic_df['Alone']


# In[70]:


# Look for >0 or ==0 to set alone status
titanic_df['Alone'].loc[titanic_df['Alone'] >0] = 'Group'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'


# In[71]:


titanic_df.head()


# In[75]:


sns.countplot('Alone',data=titanic_df,palette='Blues')


# In[77]:


# Let's start by creating a new column for legibility purposes through mapping (Lec 36)
titanic_df["Survivor"] = titanic_df.Survived.map({0: "no", 1: "yes"})

# Let's just get a quick overall view of survied vs died. 
sns.countplot('Survivor',data=titanic_df,palette='Set1')


# In[79]:


sns.factorplot('Pclass','Survived',data=titanic_df)


# In[81]:


sns.factorplot('Pclass','Survived',hue='person',data=titanic_df)


# In[83]:


sns.lmplot('Age','Survived',data=titanic_df)


# In[85]:


sns.lmplot('Age','Survived',data=titanic_df, hue='Pclass',palette='winter')


# In[86]:


generations=[10,20,40,60,80]
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generations)


# In[88]:


sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)


# In[104]:


titanic_df['Deck'] = cabin_df['Cabin']
titanic_df.head(25)


# In[101]:





# In[ ]:




