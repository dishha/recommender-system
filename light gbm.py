
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
import lightgbm as lgb


# In[2]:


import os
os.chdir(r'C:\Users\hp\Desktop\disha\churn prediction')


# In[3]:


train = pd.read_csv(r"C:\Users\hp\Desktop\disha\churn prediction\train.csv")
test = pd.read_csv(r"C:\Users\hp\Desktop\disha\churn prediction\test.csv")
songs = pd.read_csv(r"C:\Users\hp\Desktop\disha\churn prediction\songs.csv")
members = pd.read_csv(r"C:\Users\hp\Desktop\disha\churn prediction\members.csv")
songs_e = pd.read_csv(r"C:\Users\hp\Desktop\disha\churn prediction\song_extra_info.csv")


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


songs.head()


# In[7]:


members['registration_init_time'] = members['registration_init_time'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
members['expiration_date'] = members['expiration_date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))


# In[8]:


members['m_day'] = members['expiration_date'] - members['registration_init_time'] 


# In[9]:


members.head()


# In[10]:


train = pd.merge(train,songs,on ='song_id', how ='left')


# In[11]:


train = pd.merge(train,members,on ='msno', how ='left')


# In[12]:


test = pd.merge(test,songs,on ='song_id', how ='left')
test = pd.merge(test,members,on ='msno', how ='left')


# In[13]:


#no of unique members
print(len(train['msno'].unique()))

#no of unique songs
print(len(train['song_id'].unique()))

print(train.shape)


# In[14]:


train.head(4)


# In[15]:


genres_occurence = songs.genre_ids.value_counts()


# In[16]:


print(len(songs['genre_ids'].unique()))
print(len(songs['artist_name'].unique()))
print(len(songs['language'].unique()))


# In[17]:


songs.isnull().sum()


# In[18]:


def genre_id_count(x):
    if x == 'no_genre_id':
        return 0
    else:
        return x.count('|') + 1

train['genre_ids'].fillna('no_genre_id',inplace=True)
test['genre_ids'].fillna('no_genre_id',inplace=True)
train['genre_ids_count'] = train['genre_ids'].apply(genre_id_count).astype(np.int8)
test['genre_ids_count'] = test['genre_ids'].apply(genre_id_count).astype(np.int8)


# In[ ]:


for f in train.columns: 
    if train[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder() 
        lbl.fit(list(train[f].values)) 
        train[f] = lbl.transform(list(train[f].values))

for f in test.columns: 
    if test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder() 
        lbl.fit(list(test[f].values)) 
        test[f] = lbl.transform(list(test[f].values))

train.fillna((-999), inplace=True) 
test.fillna((-999), inplace=True)

train=np.array(train) 
test=np.array(test) 
train = train.astype(float) 
test = test.astype(float)


# In[ ]:


train_X = train.drop(['target'], axis =1)
train_y = train['target'].values

test_X = test.drop(['id'],axis=1)
test_y = test['id'].values


# In[ ]:


#model training

d_train = lgb.Dataset(train_X, label=train_y)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10
clf = lgb.train(params, d_train, 100)


# In[ ]:


y_pred=clf.predict(test_X)
#convert into binary values
for i in range(0,99):
    if y_pred[i]>=.5:       # setting threshold to .5
       y_pred[i]=1
    else:  
       y_pred[i]=0

