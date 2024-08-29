import pandas as pd
import string
import nltk
import nltk
from nltk.tokenize import word_tokenize
import seaborn as sns
import pandas as pd 

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

column_name = ['game_name','sentiment','comment']
data = pd.read_csv('/workspaces/August/data/twitter_training.csv',names= column_name, header=None)


data = data.drop(['sentiment'], axis=1)


data = data.dropna(subset=['comment'])

import nltk
from nltk.tokenize import word_tokenize

# will lower case all words and delete annesesery signs

data['clean_comment'] = data['comment'].str.translate(str.maketrans('','',string.punctuation)).str.lower()

# apply function from liberty nltk for a purpose of tokenezation

data['clean_comment_tokenized'] = data['clean_comment'].apply(word_tokenize)

english_stopwords = set(stopwords.words('english'))

def remove_stopwords(tokens):
  return [word for word in tokens if word.lower() not in english_stopwords]

data['clean_tokens'] = data['clean_comment_tokenized'].apply(remove_stopwords)

data['clean_commnet'] = data['clean_tokens'].apply(lambda x: ' '.join(x))

# Flattening the List: The list comprehension [token for sublist in data['clean_tokens'] for token in sublist] flattens the list of token lists into a single list of tokens.
# Frequency Distribution: FreqDist(all_tokens) is then used to create a frequency distribution from this flattened list.
# Most Common Tokens: freq_dist.most_common(20) retrieves the 20 most common tokens

all_tokens = [token for sublist in data['clean_tokens'] for token in sublist ]

freq_dist = FreqDist(all_tokens)

print(freq_dist)

data['Pos_tokens'] = data.clean_tokens.apply(nltk.pos_tag)

def extract_adj(tokens):
  adjectives = []
  for x in tokens:
    if x[1] not in ['NN','JJ','VB']:
      adjectives.append(x[0])
  return adjectives

from wordcloud import WordCloud

data['adjectives'] = data['Pos_tokens'].apply(extract_adj)

data['clean_adjectives'] = data['adjectives'].apply(lambda x: ' '.join(x))

sent = SentimentIntensityAnalyzer()
def polarity_score(review):
    # Initializing the sentiment Analyzer


    # Extracting the sentiment polarity scores of a review
    score = sent.polarity_scores(review)

    # Getting the compound score
    compound = score['compound']

    if compound > 0.05:
        return 'positive'
    elif compound < -0.5:
        return 'negative'
    else:
        return 'neutral'
data['sentiment'] = data.clean_adjectives.apply(polarity_score)

print(data.head(5))

data.to_csv('/workspaces/August/data/twiter_model.csv', sep=',')