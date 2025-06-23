import pandas as pd

models = ['RF (full)', 'RF (top 40)', 'MLR (full)', 'MLR (top 40)', 
          'XGBoost (full)', 'XGBoost (top 40)', 'SVM (full)', 'SVM (top 40)', 'MLP (full)', 'MLP (top 40)']
phases = ['Overall', 'G0', 'G1', 'G2', 'S']

# Create a DataFrame with models as rows and accuracy metrics as columns
results_df = pd.DataFrame(index=models, columns=phases)

# Save to file
results_df.to_csv("classification_results.csv")