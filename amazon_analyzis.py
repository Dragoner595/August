import pandas as pd 
import numpy as np 

data = pd.read_csv('/workspaces/August/data/amazon_reviews.csv')

data = data.drop(['date'],axis = 1)

print(data.info())
print(data.describe())

# Will create a new feature for dataset wich will have a length of verified review 
# len function didnt worked without filling the nan value 
data['verified_reviews'] = data['verified_reviews'].fillna('')

data['length'] = data['verified_reviews'].apply(len)

print(data.head(5))
print('Worked till 1')
# We want to read review from list so we select review what we want and reed it  

#print(data[data['length'] >= 2500]['verified_reviews'].iloc[0])

average = data['length'].mean()

# Find the row with the length closest to the average
closest_index = (data['length'] - average).abs().idxmin()

#print(data.loc[closest_index]['verified_reviews'])

positive = data[data['feedback'] == 1 ]

negative =data[data['feedback'] == 0]

sentences = positive['verified_reviews'].tolist()

print('Worked till 2')


import string
import nltk
from nltk.corpus import stopwords

# Ensure 'feedback' is the correct column name
negative = data[data['feedback'] == 0]

# Example to demonstrate punctuation and stopwords removal
nltk.download('stopwords')

mini_challenge = 'Here is a mini challenge, that will teach you how to remove stopwords and punctuation'

# Remove punctuation
challenge_no_punctuation = ''.join([char for char in mini_challenge if char not in string.punctuation])

# Remove stopwords
challenge_clean = [word for word in challenge_no_punctuation.split() if word.lower() not in stopwords.words('english')]

# Output the cleaned listfrom sklearn.feature_extraction.text import CountVectorizer
print('Worked till 3')
from sklearn.feature_extraction.text import CountVectorizer

sample_data = ['This is the first paper.','This document is the second paper.','And this is the third one.']

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(sample_data)

# showing us unique sentences 
print(vectorizer.get_feature_names_out())
print('Worked till 4')

def message_cleaning(message):
    test_punk_removed = [char for char in message if char not in string.punctuation]
    test_punk_removed_join = ''.join(test_punk_removed)
    test_punk_removed_clean = [word for word in test_punk_removed_join.split() if word.lower() not in stopwords.words('english')]
    return test_punk_removed_clean

data['clean_tokens'] = data['verified_reviews'].apply(message_cleaning)

print(data)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer= message_cleaning)
reviews_countvectorizer = vectorizer.fit_transform(data['verified_reviews'])

print(vectorizer.get_feature_names_out)
print(reviews_countvectorizer.toarray())
reviews = pd.DataFrame(reviews_countvectorizer.toarray())

X = reviews
y = data['feedback']

from sklearn.model_selection import train_test_split 

X_train, X_test ,y_train , y_test = train_test_split(X,y,test_size = 0.2)

from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train,y_train)

from sklearn.metrics import classification_report, confusion_matrix 

y_predict_test = NB_classifier.predict(X_test)

cm = comfusion_matrix(y_test,y_predict_test)

sns.heatmap(cm,annot = True)

