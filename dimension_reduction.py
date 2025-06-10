import pandas as pd 
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

cell_cycle = pd.read_csv('cell_cycle_tidied.csv') 
features = cell_cycle.drop(['phase', 'age', 'PHATE_1', 'PHATE_2'], axis=1)
labels = cell_cycle['phase']

tsne = TSNE(n_components=2, random_state=949, perplexity=100, max_iter=1000)
tsne_results = tsne.fit_transform(features)

phase_to_color = {'G0': 0, 'G1': 1, 'S': 2, 'G2': 3, 'M': 4}
colors = labels.map(phase_to_color)

# Create a scatter plot of the t-SNE results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors,
                      cmap='tab10', alpha=0.6)
plt.title('t-SNE of Cell Cycle Data')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
#plt.colorbar(scatter, ticks=['G0', 'G1', 'S', 'G2', 'M'], label='Phase')
plt.clim(-0.5, 3.5)
plt.grid(True)
plt.tight_layout()
#plt.show()

# Save the plot
plot_path = "tsne_100.png"
plt.savefig(plot_path, dpi=300)
plt.show()

plot_path