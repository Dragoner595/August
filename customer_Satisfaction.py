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

#print(nltk.pos_tag(data.product_review_tokenized[0]))

nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')

data['POS_tokens'] = data.product_review_tokenized.apply(nltk.pos_tag)

def extaract_adj(tokens):
    adjectives = []
    for x in tokens:
        if x[1] in ['JJ','JJR','JJS']:
            adjectives.append(x[0])
    return adjectives

data['adjectives'] = data.POS_tokens.apply(extaract_adj)

print(data.head())


adj_tops = ''

for x in data[data.product_category == 'Tops'].adjectives:
    adj_tops += ' '.join(x) + ' '

print(adj_tops)

word_cloud = WordCloud(width = 800, height = 600 , background_color = 'white').generate(adj_tops)

#def visualization_adjectives(category):
    #adjectives = ""

    #for x in data[data.product_category == category].adjectives:
       #adjectives += "".join() + " "

    #word_cloud = WordCloud(width = 800 , height = 600 , background_color= 'white').generate(adjectives)
    #plt.imshow(word_cloud)
    #plt.axis('off')
    #plt.show()

#visualization_adjectives('Jackets')

sent = SentimentIntensityAnalyzer()
review = data.product_review_cleaned[0]
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

def polarity_score(review):
    # Initilizing the sentiment Analyzer 
    sent = SentimentIntensityAnalyzer()

    # Extracting the sentiment polarity  scores of a review 
    score = sent.polarity_scores(review)

    #Getting the compound score
    compound = scores['compound']

    if compound > 0.05:
        return 'positive'
    elif compound < -0.5:
        return 'negative'
    else:
        return 'neutral'
    
print(polarity_score("This product is amazing the qality is really good"))
data['sentiment'] = data.product_review_cleaned.apply(polarity_score)

print(data.head(5))
df = data.groupby(['product_category', 'sentiment']).size().reset_index(name='counts')

px.bar(df,x = 'product_category', y = 'counts', color = 'sentiment' , barmode = 'group')

