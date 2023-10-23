#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('Algerian_forest_fires_dataset_UPDATE.csv',header=1 )
df.head()


# In[3]:


df.shape


# In[4]:


df.drop([122,123],inplace=True)
df.reset_index(inplace=True)
df.drop('index',axis=1,inplace=True)


# In[5]:


df.shape


# In[6]:


df.tail(124)


# In[7]:


df.columns


# In[8]:


df.columns=[i.strip() for i in df.columns]
df.columns


# In[9]:


for feature in [ 'DC']:
    df[feature]=df[feature].str.replace(" ","")    
    


# In[10]:


for feature in [ 'Classes']:
    df[feature]=df[feature].str.replace(" ","")    


# In[11]:


df['Classes']  


# In[12]:


df['FWI'].unique()


# In[13]:


df[df['FWI']=='fire   '].index


# In[14]:


df['FWI' ].mode()


# In[15]:


df.loc[165,'FWI']='0.4'


# In[16]:


df.info() 


# In[17]:


df.isnull().sum()


# In[18]:


df[df['Classes']=='nan'].index


# In[19]:


df.loc[165,'Classes']='fire'


# In[20]:


df['Classes'].unique()


# In[21]:


df['Classes']


# In[22]:


dfs=  { 'notfire':0,'fire':1}
df['Classes']= df['Classes'].map(dfs)


# In[23]:


df['Classes']
df['Classes'].unique()


# In[24]:


df['day']=df['day'].astype(int)
df['month']=df['month'].astype(int)
df['year']=df['year'].astype(int)
df['Temperature']=df['Temperature'].astype(int)
df['RH']=df['RH'].astype(int)
df['Ws']=df['Ws'].astype(int)
df['Rain']=df['Rain'].astype(float)
df['FFMC']=df['FFMC'].astype(float)
df['DMC']=df['DMC'].astype(float)
df['ISI']=df['ISI'].astype(float)
df['BUI']=df['BUI'].astype(float)
df['DC']=df['DC'].astype(float)
df['FWI']=df['FWI'].astype(float)
df['Classes']=df['Classes'].astype(int)


# In[25]:


data=df.copy()
data.head()


# In[26]:


df.info()


# In[27]:


data.describe()


# In[28]:


data.cov()


# In[29]:


plt.figure(figsize=(20,40), facecolor='white')
plotnumber = 1

for column in data:
    if plotnumber<=15 :     
        ax = plt.subplot(8,2,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize=20)
        
    plotnumber+=1
plt.show() 


# In[30]:


plt.figure(figsize=(20,15))
sns.heatmap(data.corr(),cmap='CMRmap',annot=True)


# In[31]:


data.Classes.value_counts()


# In[32]:


plt.subplots(figsize=(20,10))
sns.histplot('distribution of temperature',x=data.Temperature,color='b',kde=True)
plt.title('Temperature distribution',weight='bold',fontsize='30',pad=20 )


# In[33]:


import matplotlib
matplotlib.rcParams['figure.figsize']=(20,10)
sns.barplot(x='Temperature',y='Classes',data=data)


# In[34]:


num_col=[feature for feature in data.columns if data[feature].dtype!='0']
num_col


# In[35]:


def outliers(data,Temp):
    iqr=1.5*(np.percentile(data[Temp],75)-np.percentile(data[Temp],25))
    data.drop(data[data[Temp]>(iqr+np.percentile(data[Temp],75))].index,inplace=True)
    data.drop(data[data[Temp]<(np.percentile(data[Temp],25)-iqr)].index,inplace=True)
outliers(data,'Temperature')   


# In[36]:


sns.boxplot(data.Temperature )


# In[37]:


outliers(data,'Ws')       
sns.boxplot(data.Ws )


# In[38]:


outliers(data,'Rain')   
sns.boxplot(data.Rain )


# In[39]:


outliers(data,'FFMC')   
sns.boxplot(data.FFMC )


# In[40]:


outliers(data,'DMC')   
sns.boxplot(data.DMC )


# In[41]:


outliers(data,'DC')   
sns.boxplot(data.DC )


# In[42]:


outliers(data,'ISI')   
sns.boxplot(data.ISI )


# In[43]:


outliers(data,'BUI')   
sns.boxplot(data.BUI )


# In[44]:


outliers(data,'FWI')   
sns.boxplot(data.FWI )


# In[45]:


X = data.drop(columns = ['Classes'])
y = data['Classes']


# In[46]:


print(X)


# In[47]:


print(y)


# #Random Forest

# In[48]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[49]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[50]:


param_grid={
    'n_estimators':[100,200,300],
    'max_depth':[5,10,15],
    'min_samples_split':[2,5,10]
}


# In[51]:


rf=RandomForestClassifier()
grid_search=GridSearchCV(rf,param_grid,cv=5)
grid_search.fit(X,y)


# In[52]:


print('best param:',grid_search.best_params_)
print('best score:',grid_search.best_score_)


# In[53]:


RF=RandomForestClassifier(max_depth=5,min_samples_split=5,n_estimators=200)


# In[54]:


RF.fit(X,y)
y_pred=RF.predict(X_test)
y_pred


# In[55]:


print("X_test: \n", X_test)
print("y_pred: \n", y_pred)


# In[56]:


#Accuracy
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[57]:


from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

# Initialize the base classifier
base_clf = RandomForestClassifier(random_state=42)

# Initialize the bagging classifier with desired hyperparameters
bagging = BaggingClassifier(base_estimator=base_clf, n_estimators=200, random_state=42)

# Fit the bagging classifier to the training data
bagging.fit(X_train, y_train)

# Predict the class labels on the test data
y_pred = bagging.predict(X_test)


# In[58]:


#Accuracy
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[59]:


from sklearn.metrics import precision_score

# Calculate precision on the test data
precision = precision_score(y_test, y_pred)

# Print the precision score
print("Precision:", precision)


# In[60]:


from sklearn.metrics import f1_score

# Calculate the F1-score on the test data
f1 = f1_score(y_test, y_pred)

# Print the F1-score
print("F1-score:", f1)


# In[61]:


import pickle
file='finalmodelrf.sav'
pickle.dump(RF,open(file,'wb'))


# In[62]:


model=pickle.load(open('finalmodel.sav','rb'))


# In[63]:


X1=[[3,2,2023,24,]]
model.predict(X1)


# In[ ]:


f='Randomforest.pkl'
pickle.dump(RF,open(f,'wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




