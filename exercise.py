import pandas as pd
import string 

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords 

data = pd.read_csv('data/reviews.csv')
print(data.head())

# write your code here 

data.Review = data.Review.str.translate(str.maketrans('','',string.punctuation))

# tokenezation and removing the stopwords 

english_stopwords = stopwords.words('english')


def tokenize(review):
    review = str(review)
    tokens = nltk.word_tokenize(review)
    tokens = [t for t in tokens if t not in english_stopwords]
    return  tokens 

data['tokens'] = data.Review.apply(tokenize)
[print(data.head())]

