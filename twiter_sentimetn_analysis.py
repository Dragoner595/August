import pandas as pd
import string
import nltk
import nltk
from nltk.tokenize import word_tokenize
import seaborn as sns


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
data = pd.read_csv('/content/sample_data/twitter_training.csv',names= column_name, header=None)


# will drop column with sentiments from other person analysis

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
    if x[1] in x[1]:
      adjectives.append(x[0])
  return adjectives

from wordcloud import WordCloud

data['adjectives'] = data['Pos_tokens'].apply(extract_adj)

print(data.head(5))

adj = ''

for x in data.adjectives:
    adj += ' '.join(x) + ' '

wordcloud = WordCloud()

# Generate the word cloud
wordcloud = wordcloud.generate(adj)

# Plot the word cloud
plt.figure(figsize=(20, 20))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Turn off axis lines and labels
plt.show()


