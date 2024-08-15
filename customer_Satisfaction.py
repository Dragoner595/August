import pandas as pd
import string 
import nltk

# Importing NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('average_perceptron_tagger')
nltk.download('vader_lexicon')

from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud 
import plotly.express as px 
import matplotlib.pyplot as plt 

print("everything imported")

data = pd.read_csv('data/dataset.csv')

print(data.product_category.value_counts())
print(data.info())

data.product_review = data.product_review.str.translate(str.maketrans('', '', string.punctuation)).str.lower()

# Tokenizing the reviews
data['product_review_tokenized'] = data.product_review.apply(nltk.word_tokenize)

# Removing stopwords
english_stopwords = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in english_stopwords]

data['cleaned_tokens'] = data['product_review_tokenized'].apply(remove_stopwords)

# Convert cleaned tokens back to a single string if needed for further processing
data['product_review_cleaned'] = data['cleaned_tokens'].apply(lambda x: " ".join(x))

print(data.info())
print(data.head())

# Filter the DataFrame to get rows where 'product_category' is 'Tops'
data_tops = data[data['product_category'] == 'Tops']

# Flatten the 'cleaned_tokens' list for 'Tops' category
tops_tokens = [token for sublist in data_tops['cleaned_tokens'].tolist() for token in sublist]

# Flatten the 'product_review_tokenized' list for 'Tops' category
down_tokens = [token for sublist in data_tops['product_review_tokenized'].tolist() for token in sublist]

# Frequency distribution of tokens
freq_dist = FreqDist(tops_tokens)
freq_dist_down = FreqDist(down_tokens)

# Print most common tokens
print(freq_dist.most_common(20))
print(freq_dist_down.most_common(20))

print(nltk.pos_tag(data.product_review_tokenized[0]))
nltk.dowload('tagsets')
nltk.help.upenn_tagset()

def extract_adj(tokens):
    adjectives = []
    for x in tokens:
        if x[1] in ['JJ','JJR','JJS']:
            adjectives.append(x[0])
    return adjectives

data['adjectives'] = data.POS_tokens.apply(extract_adj)

print(data.head())

