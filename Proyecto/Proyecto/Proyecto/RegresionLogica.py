import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_matrix, accuracy_score, precision_score, recall_score, f1_score

# Cargar los datos desde un archivo CSV
data = pd.read_csv('/datasets/data.csv')

# Explorar el conjunto de datos
print(data.head())
print(data.info()) 
print(data.describe())