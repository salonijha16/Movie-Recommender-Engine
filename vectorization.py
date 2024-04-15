#!/usr/bin/env python
# coding: utf-8

# In[126]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter


# In[20]:


movies=pd.read_csv("tmdb_5000_movies.csv")
credits=pd.read_csv("tmdb_5000_credits.csv ")


# In[21]:


movies.head()


# In[22]:


credits.head()


# In[23]:


credits.head()


# In[24]:


credits.head(1)["cast"].values


# In[25]:


credits.head(1)["crew"].values


# In[26]:


movies.merge(credits,on='title')


# In[39]:


movies = movies.merge(credits,on='title')


# In[40]:


credits.head(1)


# 

# In[41]:


movies.head(1)


# In[35]:


movies.head(1)


# In[12]:


credits.head()


# In[49]:


movies= movies[['id','title','overview','genres','keywords','cast','crew']]


# In[47]:


movies.head()


# In[44]:


movies['original_language'].value


# In[50]:


movies['original_language'].value_counts()


# In[51]:


movies.info()


# In[52]:


movies.head()


# In[53]:


movies.isnull().sum()


# In[54]:


movies.dropna(inplace=True)


# In[55]:


movies.duplicate()


# In[56]:


movies.duplicated().sum()


# In[57]:


movies.iloc[0].genres


# In[58]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L    
        


# In[ ]:





# In[59]:


convert('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[60]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]') 


# In[61]:


movies['genres']=movies['genres'].apply(convert)


# In[62]:


movies.head()


# In[63]:


movies['keywords']=movies['keywords'].apply(convert)


# In[64]:


movies.head()


# In[65]:


movies['cast'][0]


# In[66]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            
            L.append(i['name'])
            counter+=1
        else:
            break    
    return L    
        


# In[67]:


movies['cast'].apply(convert3)


# In[68]:


movies['cast']=movies['cast'].apply(convert3)


# In[69]:


movies.head()


# In[70]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            
            L.append(i['name'])
            break
    return L    


# In[71]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[72]:


movies.head()


# In[73]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[74]:


movies.head()


# In[75]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[76]:


movies.head()


# In[77]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[78]:


movies.head()


# In[80]:


new_df=movies[['id','title','tags']]


# In[81]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[82]:


new_df.head()


# In[134]:


import nltk


# In[135]:


get_ipython().system('pip install nltk')


# In[136]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[141]:


def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)    


# In[144]:


new_df['tags']=new_df['tags'].apply(stem)


# In[139]:


new_df['tags'][0]


# In[83]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[84]:


new_df.head()


# In[145]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[146]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[147]:


vectors


# In[148]:


vectors[0]


# In[149]:


cv.get_feature_names()


# In[150]:


ps.stem('loved')


# In[151]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')


# In[153]:


from sklearn.metrics.pairwise import cosine_similarity


# In[165]:


similarity=cosine_similarity(vectors)


# In[168]:


similarity[1]


# In[ ]:




