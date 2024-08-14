import pandas as pd
import string 
import nltk

# importing the NLTK english stop words
nltk.download('stopwords')

# download the nltk sentence tokenizer 
nltk.download('punkt')

# downloading the nltk pos tagger 
nltk.download('average_perceptron_tagger')

# dowloading the nltk vader laxicon 
nltk.download('vader_lexicon')

# Importing the NLTK english sto pwords 

from nltk.corpus import stopwords

# Import frequency distribution from nltk 
from nltk.probability import FreqDist

#importing Vader dictionary. It is a rule - base seniment analyser 

from nltk.sentiment import SentimentIntensityAnalyzer

# Importing data visualization modules 

from wordcloud import WordCloud 
import plotly.express as px 
import matplotlib.pyplot as plt 

print("everything imported")

data = pd.read_csv('data/dataset.csv')

print(data.product_category.value_counts())
print(data.info())

data.product_review = data.product_review.str.translate(str.maketrans('', '', string.punctuation)).str.lower()

#print(data.head(1))

nltk.word_tokenize('This is a sentence please tokenize me')

# Tokenization is the process of breaking down a continuous stream of text, such as a sentence or a paragraph, into smaller units called tokens.These tokens typically correspond to words but can also represent subword units like prefuxes, suffixes and stems
# Tokenization facilitates the transformation of text into a format that machine learning algorithms can understand 
# nltk.word_tokenize(This is a sentence please tokenize me)

tokens = nltk.word_tokenize(data.product_review[0])

#print(tokens)

# Stop words are common words that appear frequently in a language and have little semantic value.Removing them is essential in natural language processing tasks to reduce data size speed up processing and improve the accuracy of algorithms by focusing on more informative words that convey the actual meaning of a text.

english_stopwords = stopwords.words('english')
#english_stopwords.extend(['im','its','youre','every','thing','cant','dont','doesnt'])
#print(english_stopwords)

data['product_review_tokenized'] = data.product_review.apply(nltk.word_tokenize)
print(data.info())
# in output we get some stop tokens so we will try to remvoe them 

data['product_review_tokenized']= data['product_review_tokenized'].drop(english_stopwords)

def remove_stopwords(tokens):
    return(x for x in tokens if x in english_stopwords)

data['cleaned_tokens'] = data.product_review_tokenized.apply(remove_stopwords)

data['product_review_cleaned'] = data.cleaned_tokens.apply(lambda x: " ".join(x))

print(data)

data.product_category.value_counts()


# Filter the DataFrame to get rows where 'product_category' is 'Tops'
data_tops = data[data['product_category'] == 'Tops']

#print(data)
#print(data_tops.head(5))
# Access the 'cleaned_tokens' column from the filtered data
# Flatten the 'cleaned_tokens' list for 'Tops' category

tops_tokens = [token for sublist in data_tops['cleaned_tokens'].tolist() for token in sublist]

# Flatten the 'product_review_tokenized' list for 'Tops' category
down_tokens = [token for sublist in data_tops['product_review_tokenized'].tolist() for token in sublist]


freq_dist = FreqDist(tops_tokens)
freq_dist_down = FreqDist(down_tokens)

#print(freq_dist.most_common(20))

print(freq_dist.most_common(20))
[('the', 2000), ('i', 1663), ('and', 1446), ('it', 1318), ('a', 1206), ('is', 979), ('this', 855), ('to', 692), ('in', 569), ('with', 466), ('but', 397), ('on', 382), ('of', 357), ('so', 352), ('for', 348), ('love', 337), ('top', 334), ('my', 266), ('that', 260), ('its', 248)]

print(data.product_review[0])




