import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

cell_cycle = pd.read_csv('/Users/mariahloehr/IICD/IICD/Data/cell_cycle_tidied.csv') 
x = cell_cycle["PHATE_1"]
y = cell_cycle["PHATE_2"]
labels = cell_cycle['phase']

rf_loco = pd.read_csv('/Users/mariahloehr/IICD/IICD/LOCO/rf_loco_full.csv')
xgb_loco = pd.read_csv('/Users/mariahloehr/IICD/IICD/LOCO/xgb_loco_full.csv')
mlp_loco = pd.read_csv('/Users/mariahloehr/IICD/IICD/LOCO/mlp_loco_full.csv')

reference_row = rf_loco.loc[["feature"]]  # Keep as DataFrame (not Series)

# Drop the reference row from each DataFrame
df1_no_ref = rf_loco.drop("feature")
df2_no_ref = xgb_loco.drop("feature")
df3_no_ref = mlp_loco.drop("feature")

# Add the remaining dataframes together
summed = df1_no_ref + df2_no_ref + df3_no_ref

# Add the reference row back
merged = pd.concat([reference_row, summed])

# Optional: sort if needed
merged = merged.sort_index()

phase_to_color = {'G0': 0, 'G1': 1, 'S': 2, 'G2': 3, 'M': 4}
colors = labels.map(phase_to_color)

# Create a scatter plot of the t-SNE results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(x, y, c=colors,
                      cmap='tab10', alpha=0.6)
plt.title('PHATE of Cell Cycle Data')
plt.xlabel('PHATE 1')
plt.ylabel('PHATE 2')
#plt.colorbar(scatter, ticks=['G0', 'G1', 'S', 'G2', 'M'], label='Phase')
plt.clim(-0.5, 3.5)
plt.grid(True)
plt.tight_layout()
#plt.show()

# Save the plot
plot_path = "phate.png"
plt.savefig(plot_path, dpi=300)
plt.show()

plot_path