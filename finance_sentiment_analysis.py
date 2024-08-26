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