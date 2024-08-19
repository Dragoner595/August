import pandas as pd 
import string 

# Importing Natural Language Processing toolkit 
import nltk

# Download the NLTK english stop words 

nltk.download('stopwords')

# Importing data visualization modules 

from wordcloud import WordCloud

print('Modules are imported')

# Loading the dataset 

data = pd.read_csv('/workspaces/August/data/reviews.csv')

# review to lower case 

data.Review = data.Review.str.lower()

# removing the punctuation

data.Review = data.Review.str.translate(str.maketrans("","",string.punctuation))

# toekenezation

data['tokens'] = data.Review.apply(nltk.word_tokenize)

print(data.head())