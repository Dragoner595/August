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

print(data.head(1))