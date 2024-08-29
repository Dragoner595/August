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
data = pd.read_csv('/workspaces/August/data/twitter_training.csv',names= column_name, header=None)


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

sent = SentimentIntensityAnalyzer()
review = data['clean_commnet'].iloc[0]
print(review)

scores = sent.polarity_scores(review)
print(scores)

# POS: The probability of positive sentiment
# Neu: The probability of neutral sentiment
# Neg The probability of negative sentiment
# compound: The normalized compound score that takes values from -1 to 1

# we can use the compound score to find sentiment of each review
# if compound score >= 0.05 then positive
# if compound score >= -0.05 and 0.05 then neutral
# if compound score <= -0.05 then negative

sent = SentimentIntensityAnalyzer()
data['clean_adjectives'] = data['adjectives'].apply(lambda x: ' '.join(x))

def polarity_score(review):
  # Initilizing the sentiment Analyzer
  # Extracting the sentimetn polarity scores of a review
  score = sent.polarity_scores(review)

  #getting the compound score
  compound = scores['compound']

  if compound > 0.05:
    return 'positive'
  elif compound < -0.5:
    return 'negative'
  else:
    return 'neutral'

#data['sentiment'] = data.clean_commnet.apply(polarity_score)


data['sentiment'] = data.clean_adjectives.apply(polarity_score)


# SO after analizis i understand that someone corupted data set with spesific tokens wich create a negative outlook if you do sentiment analizis buy yourself so will try to
# find this tokens and use all other then them to create a right dataset

Pos_tokens = [token for sublist in data['Pos_tokens'] for token in sublist ]

freq_dist1 = FreqDist(Pos_tokens)

top_100 = freq_dist1.most_common(500)
top_100
# game is review about the game i will not select tree different tokens as dead kill and murder and see how it will affect general outlook
# dead JJ = 1994  , 'NN' = im , murder

data['Pos_tokens_clean'] = data.clean_tokens.apply(nltk.pos_tag)

def extract_non_adj(tokens):
  adjectives = []
  for x in tokens:
    if x[1] not in ['NN','JJ','VB']:
      adjectives.append(x[0])
  return adjectives


data['non_adjectives'] = data['Pos_tokens_clean'].apply(extract_non_adj)

data.head(5)
# POS: The probability of positive sentiment
# Neu: The probability of neutral sentiment
# Neg The probability of negative sentiment
# compound: The normalized compound score that takes values from -1 to 1

# we can use the compound score to find sentiment of each review
# if compound score >= 0.05 then positive
# if compound score >= -0.05 and 0.05 then neutral
# if compound score <= -0.05 then negative

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

data['clean_non_adjectives'] = data['non_adjectives'].apply(lambda x: ' '.join(x))

sent = SentimentIntensityAnalyzer()
def polarity_score(review):
    # Initializing the sentiment Analyzer


    # Extracting the sentiment polarity scores of a review
    score = sent.polarity_scores(review)

    # Getting the compound score
    compound = score['compound']

    if compound > 0.05:
        return 'positive'
    elif compound < -0.05:
        return 'negative'
    else:
        return 'neutral'
#data['sentiment'] = data.clean_commnet.apply(polarity_score)


data['sentiment1'] = data.clean_non_adjectives.apply(polarity_score)

vectorizer = CountVectorizer(max_features=5000)
reviews_countvectorizer = vectorizer.fit_transform(data['clean_non_adjectives'])

print(vectorizer.get_feature_names_out())
print(reviews_countvectorizer.toarray())

reviews = pd.DataFrame(reviews_countvectorizer.toarray())
X = reviews
y = data['sentiment1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

y_predict_test = NB_classifier.predict(X_test)

cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
print(cm)