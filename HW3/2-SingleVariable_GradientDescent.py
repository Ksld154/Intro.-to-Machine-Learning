import numpy  as np
import pandas as pd 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

dataset = pd.read_csv('Concrete_Data.csv')
dataset_df = pd.DataFrame(dataset)

n_data = 1030


