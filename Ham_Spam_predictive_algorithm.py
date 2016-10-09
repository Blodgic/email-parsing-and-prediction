
# coding: utf-8

# In[1]:

#ham2 emails
from pandas import DataFrame
from pymongo import MongoClient
client = MongoClient('mongodb://192.168.67.90:27017')
client.database_names()
db = client['spam_database']
collection = db.ham_clean2
dataframe_ham_clean = DataFrame(list(collection.find()))


# In[2]:

#spam2 emails
from pandas import DataFrame
from pymongo import MongoClient
client = MongoClient('mongodb://192.168.67.90:27017')
client.database_names()
db = client['spam_database']
collection = db.spam_clean2
dataframe_spam_clean = DataFrame(list(collection.find()))


# In[3]:

len(dataframe_spam_clean)


# In[4]:

len(dataframe_ham_clean)


# In[5]:

dataframe_spam_clean.query('ContentType_body == ["text/html"]')[['Subject','ContentType_body','body','body_text_normalize']]


# In[6]:

from pandas import DataFrame
from pymongo import MongoClient
client = MongoClient('mongodb://192.168.67.90:27017')
client.database_names()
db = client['spam_database']
collection = db.spam_clean
dataframe_spam_clean = DataFrame(list(collection.find()))


# In[7]:

dataframe_ham_clean.columns


# In[8]:

dataframe_spam_clean.columns


# In[10]:

import pandas as pd
combo_clean = pd.concat([dataframe_ham_clean, dataframe_spam_clean], axis=0)
len(combo_clean)


# In[11]:

combo_clean_reset = combo_clean.reset_index()
combo_clean_reset


# In[14]:

#Drop NA's in body
import numpy as np
combo_clean_reset['body'] = combo_clean_reset.body.replace(r'', np.nan, regex=True)
combo_clean_reset = combo_clean_reset.dropna(subset= ['body']) 
len(combo_clean_reset)


# In[16]:

import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
combo_clean_reset[['ContentType_body', 'body_text_normalize', 'ham/spam']]


# In[18]:

combo_clean_reset.ContentType_body.isnull().value_counts()


# In[34]:

from pymongo import MongoClient
try:
    client = MongoClient('mongodb://192.168.67.90:27017')
    print "Connected successfully"
            
            
except pymongo.errors.ConnectionFailure, e:
    print "Could not connect to MongoDB: %s" % e
            
db = client['spam_database']
msg_spam = db.combo_trainingSet
all_spam_msg = msg_spam.insert_many(combo_clean_reset.to_dict('records'))
all_spam_msg


# In[35]:

print 'Number of spam messages:', combo_clean_reset[combo_clean_reset['ham/spam'] == 'spam']['ham/spam'].count()
print 'Number of ham messages:', combo_clean_reset[combo_clean_reset['ham/spam'] == 'ham']['ham/spam'].count()
print 'Number of total messages: %d' % len(combo_clean_reset)


# In[36]:

#pull test data back from mongo
from pandas import DataFrame
from pymongo import MongoClient
client = MongoClient('mongodb://192.168.67.90:27017')
client.database_names()
db = client['spam_database']
collection = db.combo_trainingSet
dataframe_combo_clean_test = DataFrame(list(collection.find()))
dataframe_combo_clean_test


# In[37]:

#training data 1 from machine learning book
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
shuffled_df = dataframe_combo_clean_test
shuffled_df = shuffled_df.reindex(np.random.permutation(shuffled_df.index))

shuffled_df['body_text_normalize'] = shuffled_df['body_text_normalize'].apply(str)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(shuffled_df['body_text_normalize'], shuffled_df['ham/spam'])


# In[38]:

#training data 2 *good*
#from : <http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html>
import numpy 
from sklearn.feature_extraction.text import CountVectorizer
shuffled_df['body_text_normalize'] = shuffled_df['body_text_normalize'].apply(str)
count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(shuffled_df['body_text_normalize'].values)
counts


# In[39]:

#training data 2 *good*
#from : <http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html>
from sklearn.naive_bayes import MultinomialNB
#convert to string
#dataframe_combo_clean_test = dataframe_combo_clean_test.reindex(np.random.permutation(dataframe_combo_clean_test.index))
shuffled_df['ham/spam'] = shuffled_df['ham/spam'].apply(str)
classifier = MultinomialNB()
targets = shuffled_df['ham/spam'].values
classifier.fit(counts, targets)


# In[44]:

#test out the training set 
examples = ['Please click here for a free Amazon gift card', 'The sky is blue today']
example_counts = count_vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
predictions


# In[45]:

#PipeLining
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfTransformer
examples = ['Free Viagra call today!', 'cybersecurity jobs']
shuffled_df['ham/spam'] = shuffled_df['ham/spam'].apply(str)
shuffled_df['Subject'] = shuffled_df['Subject'].apply(str)

pipeline = Pipeline([
    ('count_vectorizer',  CountVectorizer(ngram_range=(1, 2))),
    ('tfidf_transformer',  TfidfTransformer()),
    ('classifier',  BernoulliNB(binarize=0.0))])

pipeline.fit(shuffled_df['Subject'].values, shuffled_df['ham/spam'])
pipeline.predict(examples)


# In[33]:

#shuffle data
shuffled_df = dataframe_combo_clean_test
shuffled_df = shuffled_df.reindex(np.random.permutation(shuffled_df.index))
shuffled_df


# In[57]:

#SUBJECT test

#training data = combo_clean_reset
#training data target variable = combo_clean_reset['ham/spam'] 
# training data feature of body text = combo_clean_reset['body_text_normalize'].values
#test data = dataframe_combo_clean_test
#test data target variable = dataframe_combo_clean_test['ham/spam']
#test data feature = dataframe_combo_clean_test['body_text_normalize'].values

#shuffle data
shuffled_df = dataframe_combo_clean_test
shuffled_df = shuffled_df.reindex(np.random.permutation(shuffled_df.index))


shuffled_df['Subject_cleanse'] = shuffled_df['Subject_cleanse'].apply(str)

from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score

pipeline = Pipeline([
    ('count_vectorizer',  CountVectorizer(ngram_range=(1, 2))),
    ('tfidf_transformer',  TfidfTransformer()),
    ('classifier',  MultinomialNB())])

pipeline.fit(shuffled_df['Subject'].values, shuffled_df['ham/spam'])

k_fold = KFold(n=len(dataframe_combo_clean_test), n_folds=6)
scores = []
confusion = numpy.array([[0, 0], [0, 0]])
for train_indices, test_indices in k_fold:
    train_text = shuffled_df.iloc[train_indices]['Subject_cleanse'].values
    train_y = shuffled_df.iloc[train_indices]['ham/spam'].values

    test_text = shuffled_df.iloc[test_indices]['Subject_cleanse'].values
    test_y = shuffled_df.iloc[test_indices]['ham/spam'].values

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label='ham')
    scores.append(score)

print 'Number of spam messages:', shuffled_df[shuffled_df['ham/spam'] == 'spam']['ham/spam'].count()
print 'Number of ham messages:', shuffled_df[shuffled_df['ham/spam'] == 'ham']['ham/spam'].count()
print('Total emails classified:', len(dataframe_combo_clean_test))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)
plt.matshow(confusion)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[201]:

#body_text_normalize test

#training data = combo_clean_reset
#training data target variable = combo_clean_reset['ham/spam'] 
# training data feature of body text = combo_clean_reset['body_text_normalize'].values
#test data = dataframe_combo_clean_test
#test data target variable = dataframe_combo_clean_test['ham/spam']
#test data feature = dataframe_combo_clean_test['body_text_normalize'].values
import matplotlib.pyplot as plt
from sklearn import metrics
get_ipython().magic(u'matplotlib inline')

#shuffle data
shuffled_df = dataframe_combo_clean_test
shuffled_df = shuffled_df.reindex(np.random.permutation(shuffled_df.index))
shuffled_df['body_text_normalize'] = shuffled_df['body_text_normalize'].apply(str)



from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score

pipeline = Pipeline([
    ('count_vectorizer',  CountVectorizer(ngram_range=(1, 2))),
    ('tfidf_transformer',  TfidfTransformer()),
    ('classifier',  MultinomialNB())])

pipeline.fit(shuffled_df['body_text_normalize'].values, shuffled_df['ham/spam'])

k_fold = KFold(n=len(dataframe_combo_clean_test), n_folds=6)
scores = []
confusion = numpy.array([[0, 0], [0, 0]])
for train_indices, test_indices in k_fold:
    train_text = shuffled_df.iloc[train_indices]['body_text_normalize'].values
    train_y = shuffled_df.iloc[train_indices]['ham/spam'].values

    test_text = shuffled_df.iloc[test_indices]['body_text_normalize'].values
    test_y = shuffled_df.iloc[test_indices]['ham/spam'].values

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)
    prediction_prob = pipeline.predict_proba(test_text)

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label='spam')
    scores.append(score)

    
print 'Number of spam messages:', shuffled_df[shuffled_df['ham/spam'] == 'spam']['ham/spam'].count()
print 'Number of ham messages:', shuffled_df[shuffled_df['ham/spam'] == 'ham']['ham/spam'].count()
print('Total emails classified:', len(shuffled_df))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)
print "precision when predicting spam %f" % metrics.precision_score(test_y, predictions, pos_label='spam')
print "recall when predicting spam %f" % metrics.recall_score(test_y, predictions, pos_label='spam')
print "f1 score when predicting spam %f" % metrics.f1_score(test_y, predictions, pos_label='spam')
plt.matshow(confusion)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[59]:

examples = ['Click on my Amazon gift card link', 'Get your cyber security job here']
pipeline.predict(examples)


# In[196]:

train_text


# In[105]:

train_text


# In[ ]:




# In[287]:

#Combined body_text and Subject_cleanse

#training data = combo_clean_reset
#training data target variable = combo_clean_reset['ham/spam'] 
# training data feature of body text = combo_clean_reset['body_text_normalize'].values
#test data = dataframe_combo_clean_test
#test data target variable = dataframe_combo_clean_test['ham/spam']
#test data feature = dataframe_combo_clean_test['body_text_normalize'].values
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from decimal import Decimal
get_ipython().magic(u'matplotlib inline')

#shuffle data
shuffled_df = dataframe_combo_clean_test
shuffled_df = shuffled_df.reindex(np.random.permutation(shuffled_df.index))
shuffled_df['body_text_normalize'] = shuffled_df['body_text_normalize'].apply(str)
shuffled_df['Subject_cleanse'] = shuffled_df['Subject_cleanse'].apply(str)


from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score




combined_2 = shuffled_df['Subject_cleanse'] + ' '  + shuffled_df['body_text_normalize']
combined_2 = [x for x in combined_2 if x]
combined_2= np.asarray(combined_2)

pipeline = Pipeline([
    ('count_vectorizer',  CountVectorizer()),
    ('tfidf_transformer',  TfidfTransformer()),
    ('classifier',  MultinomialNB())])

pipeline.fit(combined_2, shuffled_df['ham/spam'])

k_fold = KFold(n=len(shuffled_df), n_folds=6)
scores = []
confusion = numpy.array([[0, 0], [0, 0]])
for train_indices, test_indices in k_fold:
    train_text = combined_2[train_indices]
    train_y = shuffled_df.iloc[train_indices]['ham/spam'].values

    test_text = combined_2[test_indices]
    test_y = shuffled_df.iloc[test_indices]['ham/spam'].values

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)
    prediction_prob = pipeline.predict_proba(test_text)

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label='spam')
    scores.append(score)

    
print 'Number of spam messages:', shuffled_df[shuffled_df['ham/spam'] == 'spam']['ham/spam'].count()
print 'Number of ham messages:', shuffled_df[shuffled_df['ham/spam'] == 'ham']['ham/spam'].count()
print('Total emails classified:', len(shuffled_df))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)
print "precision when predicting spam %f" % metrics.precision_score(test_y, predictions, pos_label='spam')
print "recall when predicting spam %f" % metrics.recall_score(test_y, predictions, pos_label='spam')
print "f1 score when predicting spam %f" % metrics.f1_score(test_y, predictions, pos_label='spam')
plt.matshow(confusion)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()




#for i, prediction in enumerate(predictions[:5]):
    #print 'Prediction: %s: Actual: %s:  Message: %s: ' % (prediction, test_y[i], test_text[i])


# In[322]:

import decimal
#from decimal import Decimal
D = decimal.Decimal
predicted_decimal = pipeline.predict_proba(test_text)

f10todec = np.asarray(predicted_decimal, dtype = decimal.Decimal)
f10todec[0:100]


# In[286]:

import pandas as pd



df1 = pd.DataFrame(pipeline.predict(test_text), columns=['predicted'])
df2 = pd.DataFrame(pipeline.predict_proba(test_text), columns=[['score1', 'score2']])
df3 = pd.DataFrame(test_y, columns=['actual'])
df4 = pd.DataFrame(test_text, columns=['test_text'])
frames = [df_predic_prob, df_prediction]

result = pd.concat([df1, df2, df3, df4], axis=1)
#result = pd.DataFrame(result, columns=[['predicted', 'score1', 'score2', 'actual', 'test_text']], axis=1)
result


# In[238]:

for i, prediction in enumerate(prediction_prob[:5]):
    print 'Prediction: %s: Actual: %s:  Message: %s: ' % (prediction, test_y[i], test_text[i])


# In[224]:

import numpy as np
#myarray = np.asarray(mylist)
#columns = ["body_text_normalize", "Subject_cleanse"]
#combined_2 = shuffled_df[list(columns)].values

#combined_2 = shuffled_df(list(['Subject_cleanse'].apply(str) + ' '  + shuffled_df['body_text_normalize'].apply(str)
combined_2 = shuffled_df['Subject_cleanse'] + ' '  + shuffled_df['body_text_normalize']
combined_2 = [x for x in combined_2 if x]
combined_2= np.asarray(combined_2)
combined_2.shape
#combined_2 = filter(None, combined_2)
#combined_2






# In[193]:

combined_2 = shuffled_df['Subject_cleanse'].values + ' '  + shuffled_df['body_text_normalize'].values
filter(None, combined_2.all())


# In[137]:

from sklearn.pipeline import FeatureUnion
features = shuffled_df[['body_text_normalize', 'Subject_cleanse']].values
filter(lambda x:x != '', features.all)


# In[138]:

pd.DataFrame({'a': series.str[0].str[0], 'b': series.str[0].str[1]})





# In[494]:

train_text = shuffled_df.iloc[train_indices]['body_text_normalize'].values
train_text


# In[1]:

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import preprocessing



shuffled_df['body_text_normalize'] = shuffled_df['body_text_normalize'].apply(str)
shuffled_df['Subject_cleanse'] = shuffled_df['Subject_cleanse'].apply(str)


combined_2 = shuffled_df['Subject_cleanse'] + ' '  + shuffled_df['body_text_normalize']
combined_2 = [x for x in combined_2 if x]
combined_2= np.asarray(combined_2)


X_train_raw, X_test_raw, y_train, y_test = train_test_split(combined_2, shuffled_df['ham/spam'])
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

lb = preprocessing.LabelBinarizer()
y_test = lb.fit(y_test)


classifier = LogisticRegression()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)


confusion += confusion_matrix(y_test, predictions)
score = f1_score(y_test, predictions, pos_label='spam')
scores.append(score)

print(confusion)

#print cross_val_score(classifier, X_train, y_train, cv=10, scoring='precision')
#print cross_val_score(classifier, X_train, y_train, cv=10, scoring='recall')


for i, prediction in enumerate(predictions[:5]):
    print 'Prediction: %s. Message: %s: ' % (prediction, X_test_raw[i])
predictions = classifier.predict_proba(X_test)
for i, prediction in enumerate(predictions[:5]):
    print 'Prediction: %s. Message: %s: ' % (prediction, X_test_raw[i])
    
    
    


# In[241]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import preprocessing



shuffled_df['body_text_normalize'] = shuffled_df['body_text_normalize'].apply(str)
shuffled_df['Subject_cleanse'] = shuffled_df['Subject_cleanse'].apply(str)


combined_2 = shuffled_df['Subject_cleanse'] + ' '  + shuffled_df['body_text_normalize']
combined_2 = [x for x in combined_2 if x]
combined_2= np.asarray(combined_2)


X_train_raw, X_test_raw, y_train, y_test = train_test_split(combined_2, shuffled_df['ham/spam'])
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

lb = preprocessing.LabelBinarizer()
y_test = lb.fit(y_test)


classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#print cross_val_score(classifier, X_train, y_train, cv=10, scoring='precision')
#print cross_val_score(classifier, X_train, y_train, cv=10, scoring='recall')


for i, prediction in enumerate(predictions[:5]):
    print 'Prediction: %s. Message: %s: ' % (prediction, X_test_raw[i])
predictions = classifier.predict_proba(X_test)
for i, prediction in enumerate(predictions[:5]):
    print 'Prediction: %s. Message: %s: ' % (prediction, X_test_raw[i])


# In[310]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing


X_train_raw, X_test_raw, y_train, y_test = train_test_split(shuffled_df['body_text_normalize'], shuffled_df['ham/spam'])


 
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict_proba(X_test)
false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:, 1], pos_label='spam')
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()


# In[ ]:




# In[ ]:




# In[312]:

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

pipeline = Pipeline([
('vect', TfidfVectorizer(stop_words='english')),
('clf', LogisticRegression())
])
parameters = {
'vect__max_df': (0.25, 0.5, 0.75),
'vect__stop_words': ('english', None),
'vect__max_features': (2500, 5000, 10000, None),
'vect__ngram_range': ((1, 1), (1, 2)),
'vect__use_idf': (True, False),
'vect__norm': ('l1', 'l2'),
'clf__penalty': ('l1', 'l2'),
'clf__C': (0.01, 0.1, 1, 10),
}


# In[317]:

if __name__ == "__main__":
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,verbose=1, scoring='accuracy', cv=3)


#X_train_raw, X_test_raw, y_train, y_test = train_test_split(

X, y, = shuffled_df['body_text_normalize'], shuffled_df['ham/spam']
X_train, X_test, y_train, y_test = train_test_split(X, y)
grid_search.fit(X_train, y_train)
print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print '\t%s: %r' % (param_name, best_parameters[param_name])

predictions = grid_search.predict(X_test)
print 'Accuracy:', accuracy_score(y_test, predictions, pos_label='ham')
print 'Precision:', precision_score(y_test, predictions, pos_label='ham')
print 'Recall:', recall_score(y_test, predictions, pos_label='ham')

