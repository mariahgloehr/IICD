import pandas as pd 
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

cell_cycle = pd.read_csv('cell_cycle_tidied.csv') 
features = cell_cycle.drop(['phase', 'age'], axis=1)
labels = cell_cycle['phase']

print(labels)