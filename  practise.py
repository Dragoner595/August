import pandas as pd
import string 
import nltk

# Importing NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud 
import plotly.express as px 
import matplotlib.pyplot as plt 

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

data.Review = data.Review.str.translate(str.maketrans("","",string.punctuation)).str.lower()

# toekenezation

data['tokens'] = data.Review.apply(nltk.word_tokenize)

english_stopwords = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in english_stopwords]

data['clean_tokens'] = data['tokens'].apply(remove_stopwords)

data['product_review_cleaned'] = data['clean_tokens'].apply(lambda x: " ".join(x))

freq_dist = FreqDist(data)

print(freq_dist.most_common(20))

data['POS_tokens'] = data.tokens.apply(nltk.pos_tag)

data['adjectives'] = data.POS_tokens.apply(lambda x: [token for token , tag in x if tag == 'JJ'])


