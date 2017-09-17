
# coding: utf-8

# In[1]:

# KNN Amazon Baby Review.

import pandas as pd
import numpy as np
import nltk
import string
import numpy as np
import scipy.sparse as sparse

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.neighbors import KNeighborsClassifier


# In[2]:

#reading reviews using pandas library from amazon_baby_train.csv file
reviews = pd.read_csv('amazon_baby_train.csv')
reviews.shape

# dropping observations which are incomplete
reviews = reviews.dropna()
reviews.shape

# changing the reviews into positive and negative reviews
scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: 1 if x > 3 else 0)

# printing the mean and standard deviation of ratings
print("The Mean of the Review Attribute is : ")
print(scores.mean())
print("The Standard Deviation of the Review Attribute is : ")
print(scores.std())


# In[3]:

def splitPosNeg(Summaries):
    neg = reviews.loc[Summaries['rating'] == 0]
    pos = reviews.loc[Summaries['rating'] == 1]
    return [pos,neg]    


# In[4]:

# splitting the positive and negative review and storing them in separate arrays
[pos,neg] = splitPosNeg(reviews)


# In[5]:

# Preprocessing steps

# Using lemmatizer to lemmatizze words
lemmatizer = nltk.WordNetLemmatizer()

# using stop words to remove the words which do not contribute to the sentiment
stop = stopwords.words('english')
translation = str.maketrans(string.punctuation,' '*len(string.punctuation))

def preprocessing(line):
    tokens=[]
    line = line.translate(translation)
    line = nltk.word_tokenize(line.lower())
    #print(line)
    stops = stopwords.words('english')
    stops.remove('not')
    stops.remove('no')
    line = [word for word in line if word not in stops]
    for t in line:
        stemmed = lemmatizer.lemmatize(t)
        tokens.append(stemmed)
    return ' '.join(tokens)


# In[6]:

# Storing the positive and negative reviews in separate arrays
pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done")


# In[7]:

data = pos_data + neg_data
labels = np.concatenate((pos['rating'].values,neg['rating'].values))


# In[8]:

#tokenizing each sentence from the file into words
t = []
for line in data:
    l = nltk.word_tokenize(line)
    for w in l:
        t.append(w)


# In[9]:

# Calculating the frequency dstribution of each word
word_features = nltk.FreqDist(t)
print(len(word_features))


# In[12]:

# The most common 200 words
topwords = [fpair[0] for fpair in list(word_features.most_common(200))]
print(word_features.most_common(25))


# In[13]:

#printing the top 20 most common words
word_his = pd.DataFrame(word_features.most_common(20), columns = ['words','count'])
print(word_his)


# In[14]:

# Vectorizing the top words
vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(topwords)])


# In[15]:

# Using Tfidf Transformer on the data
tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)


# In[16]:

ctr_features = vec.transform(data)
tr_features = tf_vec.transform(ctr_features)


# In[17]:

tr_features.shape


# In[18]:

tr_features = tr_features.astype('int32')
print(tr_features.dtype)


# In[19]:

# Using KNN classifier to classify the data
clf =  KNeighborsClassifier()
clf = clf.fit(tr_features, labels)


# In[21]:

lencheck= tr_features.shape
print(lencheck)


# In[24]:

num_correct = 0;
newlen = lencheck[0]-1
for ch in range(0,newlen):
    checkPrediction = clf.predict(tr_features[ch])
    if(checkPrediction == [labels[ch]]):
        num_correct = num_correct+1;
print("Number of Correct")
print(num_correct)

accuracy = (num_correct/newlen)*100;
print("Training Accuracy");
print(accuracy);


# In[31]:

#reading reviews using pandas library from amazon_baby_test.csv file
reviews = pd.read_csv('amazon_baby_test.csv')
reviews.shape

# dropping observations which are incomplete
reviews = reviews.dropna()
reviews.shape

# changing the reviews into positive and negative reviews
scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: 1 if x > 3 else 0)

# calculating the mean of reviews
scores.mean()


# In[32]:

# splitting the positive and negative review and storing them in separate arrays
[pos,neg] = splitPosNeg(reviews)


# In[33]:

# Storing the positive and negative reviews in separate arrays
pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done")


# In[34]:

# combining the positive and negative reviews
data = pos_data + neg_data
labels = np.concatenate((pos['rating'].values,neg['rating'].values))


# In[35]:


#tokenizing each sentence from the file into words
t = []
for line in data:
    l = nltk.word_tokenize(line)
    for w in l:
        t.append(w)


# In[36]:

# Calculating the frequency dstribution of each word
word_features = nltk.FreqDist(t)
print(len(word_features))


# In[37]:

# The most common 200 words
topwords = [fpair[0] for fpair in list(word_features.most_common(200))]
print(word_features.most_common(25))


# In[38]:

# Vectorizing the top words
vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(topwords)])


# In[39]:

# Using Tfidf Transformer on the data
tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)


# In[40]:

# Transforming the features using Tfidf transformer
cte_features = vec.transform(data)
te_features = tf_vec.transform(cte_features)


# In[41]:

te_features.shape


# In[43]:

num_correct = 0;
newlen = lencheck[0]-1
for ch in range(0,newlen):
    checkPrediction = clf.predict(te_features[ch])
    if(checkPrediction == [labels[ch]]):
        num_correct = num_correct+1;
print("Number of Correct")
print(num_correct)

accuracy = (num_correct/newlen)*100;
print("Testing Accuracy");
print(accuracy);


# In[ ]:



