import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow_decision_forests as tfdf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

data = sns.load_dataset('penguins')

print(data.head())

data = data.dropna()

print(data.head())
