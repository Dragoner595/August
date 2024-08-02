import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow_decision_forests as tfdf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

data = sns.load_dataset('penguins')

data = data.dropna()
# create separate label encoder for each categorical variable 
# we create label incoders to make categorical data numeric so our ML model will understand it 

special_encoder = LabelEncoder()
island_encoder = LabelEncoder()
sex_encoder = LabelEncoder()

# Encoder each categorical variable 
# fit transform metod learn categories and fit transform them into numbers 

data['species'] = special_encoder.fit_transform(data['species'])
data['island']  = island_encoder.fit_transform(data['island'])
data['sex'] = sex_encoder.fit_transform(data['sex'])

# Saving mappings for future reference or interpritation 
# useful to save this maping for future usage , for interpreting future precition 
special_maping = dict(zip(special_encoder.classes_,range(len(special_encoder.classes_))))
island_maping = dict(zip(island_encoder.classes_,range(len(island_encoder.classes_))))
sex_maping = dict(zip(sex_encoder.classes_,range(len(sex_encoder.classes_))))

# spliting data into train test sets 

train_data,test_data = train_test_split(data,test_size= 0.3,random_state = 42 )

# conver pandas dataframes into tensorflow dataset 

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_data,label = 'species')
test_df = tfdf.keras.pd_dataframe_to_tf_dataset(test_data,label = 'species')

# create and train the model 

model = tfdf.keras.RandomForestModel()
model.fit(train_ds,verbose = 2)

# Get an inspector to access models internals 
# get opportunity to gutered information about the model and features influising its predictions  
inspector = model.make_inspector()

# Display feature importance 

importance_types = inspector.variable_importances().keys()

for importance_type in importance_types:
 print('\n'+'='*30)
 print(f'Importance Type:{importance_types}')
 print('-'*30)
 importances = inspector.variable_importances()[importance_type]
 # sort importance in descending order based on importance values 
 sorted_importance = sorted(importances,key=lambda x: x[1],reverse = True)

 # Set up formating for consistant spacing in the output 
 for feature , importance in sorted_importance:
  # ensure the feature is treated as a string 
  feature_name = feature.name if isinstance(feature,tfdf.py_tree.dataspec.SimpleColumnSpec) else feature 
  print(f"{feature_name :20}{importance:.4f}")