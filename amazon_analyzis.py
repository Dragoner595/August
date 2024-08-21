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

# We want to read review from list so we select review what we want and reed it  

#print(data[data['length'] >= 2500]['verified_reviews'].iloc[0])

average = data['length'].mean()

# Find the row with the length closest to the average
closest_index = (data['length'] - average).abs().idxmin()

print(data.loc[closest_index]['verified_reviews'])

positive = data[data['feedback'] == 1 ]

negative =data[data['feedbacl'] == 0]

sentences = positive['verified_reviews'].tolist()
len(sentences)

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

# Output the cleaned list
print(challenge_clean)