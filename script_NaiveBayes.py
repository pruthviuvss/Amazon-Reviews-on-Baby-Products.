#importing all the libraries needed,
import pandas as pd
import numpy as np
import nltk
import string
#import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from nltk.corpus import stopwords
#from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

#reading in the training dataset into a variable
reviews = pd.read_csv('amazon_baby_train.csv')
#print(reviews.shape)


#removing the tuples with null values
reviews = reviews.dropna()
#print(reviews.shape)

#segregating the tuples based on rating values.
scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: 'pos' if x > 3 else 'neg')


#checking the mean and standard deviation of the ratings.
print("MEAN OF RATINGS : ",scores.mean())
print("STANDARD DEVIATION OF RATINGS : ",scores.std())


reviews.groupby('rating')['review'].count()
#plotting a graph for all the negative and postive reviews.
#reviews.groupby('rating')['review'].count().plot(kind='bar', color=['r', 'g'], title='Label Distribution',figsize=(10, 6))
#plt.show()


def splitPosNeg(Summaries):
    neg = reviews.loc[Summaries['rating'] == 'neg']
    pos = reviews.loc[Summaries['rating'] == 'pos']
    return [pos, neg]


[pos, neg] = splitPosNeg(reviews)


#processing the data removing all punctiations and lemmetizing it.
lemmatize = nltk.WordNetLemmatizer()
translation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))



def preprocessing(line):
    tokens = []
    line = line.translate(translation)
    line = nltk.word_tokenize(line.lower())
    for t in line:
        stemmed = lemmatize.lemmatize(t)
        tokens.append(stemmed)
    return ' '.join(tokens)



pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))
#print("DONE PREPROCESSING")

data = pos_data + neg_data
labels = np.concatenate((pos['rating'].values, neg['rating'].values))


[Data_train, Data_test, Train_labels, Test_labels] = train_test_split(data, labels, test_size=0.25,
                                                                      random_state=20160121, stratify=labels)

t = []
for line in data:
    l = nltk.word_tokenize(line)
    for w in l:
        t.append(w)

word_features = nltk.FreqDist(t)
print(len(word_features))

topwords = [fpair[0] for fpair in list(word_features.most_common(5000))]
print(word_features.most_common(25))

word_his = pd.DataFrame(word_features.most_common(200), columns=['words', 'count'])

vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(topwords)])
print(c_fit)

tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)

ctr_features = vec.transform(data)
tr_features = tf_vec.transform(ctr_features)

#tr_features.shape


clf = GaussianNB()
clf = clf.fit(tr_features, labels)

tfPredication = clf.predict(tr_features)
tfAccuracy = metrics.accuracy_score(tfPredication, labels)
print("Accuracy : ",tfAccuracy)

#print(metrics.classification_report(labels, tfPredication))


#---------------------------------------------------------------------------------------------
#implementing the same procedure as training data for the testing data_set to measure accuracy.
#---------------------------------------------------------------------------------------------


reviews = pd.read_csv('amazon_baby_test.csv')
reviews.shape
reviews = reviews.dropna()
reviews.shape

scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: 'pos' if x > 3 else 'neg')

#reviews.groupby('rating')['review'].count()

#reviews.groupby('rating')['review'].count().plot(kind='bar', color=['r', 'g'], title='Label Distribution',figsize=(10, 6))

[pos, neg] = splitPosNeg(reviews)

pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))

for n in neg['review']:
    neg_data.append(preprocessing(n))

#print("DONE PREPROCESSING")

data = pos_data + neg_data
labels = np.concatenate((pos['rating'].values, neg['rating'].values))


t = []
for line in data:
    l = nltk.word_tokenize(line)
    for w in l:
        t.append(w)

word_features = nltk.FreqDist(t)
print(len(word_features))

topwords = [fpair[0] for fpair in list(word_features.most_common(5002))]
print(word_features.most_common(25))

word_his = pd.DataFrame(word_features.most_common(200), columns=['words', 'count'])
len(topwords)
vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(topwords)])
tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)
cte_features = vec.transform(data)
te_features = tf_vec.transform(cte_features)
te_features.shape
tePredication = clf.predict(te_features)
teAccuracy = metrics.accuracy_score(tePredication, labels)
print("Accuracy of test : ",teAccuracy*100,"%")
#print(metrics.classification_report(labels, tePredication))





