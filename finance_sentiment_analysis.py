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


data = pd.read_csv('/content/sample_data/finensial_statments.csv')
data.head(5)

# other in google colab will put it tomorrow 

# will lower case all words and delete annesesery signs
data['news_reviews'] = data['Sentence'].str.translate(str.maketrans('', '', string.punctuation)).str.lower()
# apply function from liberary nltk for a purpose of toekenization
data['news_reviews_tokenized'] = data['news_reviews'].apply(nltk.word_tokenize)

data.head(5)

# uploading english stopwords from liberary for deleting annesesary usual words from sentences

english_stopwords = set(stopwords.words('english'))

# function to delete all words from our column wich is similar in english stop word list
def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in english_stopwords]

data['clean_tokens'] = data['news_reviews_tokenized'].apply(remove_stopwords)

print(data.head(5))

# we will prepera new coum for future analysis

data['Sentence_cleaned'] = data['clean_tokens'].apply(lambda x: ' '.join(x))

data.head(5)

# Flattening the List: The list comprehension [token for sublist in data['clean_tokens'] for token in sublist] flattens the list of token lists into a single list of tokens.
# Frequency Distribution: FreqDist(all_tokens) is then used to create a frequency distribution from this flattened list.
# Most Common Tokens: freq_dist.most_common(20) retrieves the 20 most common tokens

all_tokens = [token for sublist in data['clean_tokens'] for token in sublist]

freq_dist = FreqDist(all_tokens)

print(freq_dist.most_common(20))
