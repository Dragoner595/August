import pandas as pd 
import numpy as np 


data = pd.read_csv('/workspaces/August/data/amazon_reviews.csv')

data = data.drop(['date'],axis = 1)

print(data.info())
print(data.describe())

# Will create a new feature for dataset wich will have a length of verified review 
# len function didnt worked without filling the nan value 
data['verified_reviews'] = data['verified_reviews'].fillna('')

data['length'] = data['verified_reviews'].apply(len)

print(data.head(5))

# We want to read review from list so we select review what we want and reed it  

#print(data[data['length'] >= 2500]['verified_reviews'].iloc[0])

average = data['length'].mean()

# Find the row with the length closest to the average
closest_index = (data['length'] - average).abs().idxmin()

print(data.loc[closest_index]['verified_reviews'])
