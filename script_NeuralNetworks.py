import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.neural_network import MLPClassifier



reviews = pd.read_csv('amazon_baby_train_sample.csv')
reviews = reviews.dropna()
scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: '1' if x > 3 else '0')
print("Mean of review attribute is : ",scores.mean())
print("reviews grouped by their count:",reviews.groupby('rating')['review'].count())
reviews.groupby('rating')['review'].count().plot(kind='bar', color=['r', 'g'], title='Label Distribution',
                                                 figsize=(10, 6))
plt.show()

def splitPosNeg(Summaries):
    neg = reviews.loc[Summaries['rating'] == '0']
    pos = reviews.loc[Summaries['rating'] == '1']
    return [pos, neg]

print("splitting done")
[pos, neg] = splitPosNeg(reviews)
lemmatizer = nltk.WordNetLemmatizer()
stop = stopwords.words('english')
translation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))


def preprocessing(line):
    tokens = []
    line = line.translate(translation)
    line = nltk.word_tokenize(line.lower())
    stops = stopwords.words('english')
    stops.remove('not')
    stops.remove('no')
    line = [word for word in line if word not in stops]
    for t in line:
        stemmed = lemmatizer.lemmatize(t)
        tokens.append(stemmed)
    return ' '.join(tokens)


pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))
print("positive done")
for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done preproc")

data = pos_data + neg_data
labels = np.concatenate((pos['rating'].values, neg['rating'].values))

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

tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)

ctr_features = vec.transform(data)
tr_features = tf_vec.transform(ctr_features)

print(tr_features.shape)

clf = MLPClassifier(hidden_layer_sizes=200,activation='logistic')
#clf=MLPClassifier()
clf = clf.fit(tr_features, labels)
tfPredication = clf.predict(tr_features)
tfAccuracy = metrics.accuracy_score(tfPredication, labels)
print("Train data: ",tfAccuracy * 100)


#---------------------------------------------------------------------------------------------
#implementing the same procedure as training data for the testing data_set to measure accuracy.
#---------------------------------------------------------------------------------------------

reviews = pd.read_csv('amazon_baby_test_sample.csv')
reviews = reviews.dropna()
scores = reviews['rating']
reviews['rating'] = reviews['rating'].apply(lambda x: '1' if x > 3 else '0')
#scores.mean()
print("reviews grouped by count:",reviews.groupby('rating')['review'].count())
reviews.groupby('rating')['review'].count().plot(kind='bar', color=['r', 'g'], title='Label Distribution',figsize=(10, 6))
plt.show()

[pos, neg] = splitPosNeg(reviews)
pos_data = []
neg_data = []
for p in pos['review']:
    pos_data.append(preprocessing(p))
for n in neg['review']:
    neg_data.append(preprocessing(n))
print("Done preprocessing p n n test")
data = pos_data + neg_data
labels = np.concatenate((pos['rating'].values, neg['rating'].values))
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
print("len(topwords)")
vec = CountVectorizer()
c_fit = vec.fit_transform([' '.join(topwords)])
tf_vec = TfidfTransformer()
tf_fit = tf_vec.fit_transform(c_fit)
cte_features = vec.transform(data)
te_features = tf_vec.transform(cte_features)
print("te_features.shape")
tePredication = clf.predict(te_features)
teAccuracy = metrics.accuracy_score(tePredication, labels)
print("Accuracy of test data : ",teAccuracy * 100)
print(metrics.classification_report(labels, tePredication))




