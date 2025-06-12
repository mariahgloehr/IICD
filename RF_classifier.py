import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
from sklearn.model_selection import GridSearchCV

# Load data
df = pd.read_csv("cell_cycle_tidied.csv")

# Combine phase M and G2 into one class
df['phase'] = df['phase'].replace({'M': 'G2'})

# Separate features and target
X = df.drop(columns=['phase', 'age', 'PHATE_1', 'PHATE_2'])
y = df['phase']

# Split data into train and test sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=949, stratify=y)

# Train Random Forest model with 500 trees and large depth
rf = RandomForestClassifier(n_estimators=500, max_depth=50, max_features='sqrt', random_state=949)
rf.fit(X_train, y_train)

# Predict on training and test sets
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

# Output results
#print("=== Training Set ===")
#print("Overall Accuracy:", accuracy_score(y_train, y_train_pred))
#print("Per-Class Accuracy:\n", classification_report(y_train, y_train_pred, digits=3))

#print("\n=== Test Set ===")
#print("Overall Accuracy:", accuracy_score(y_test, y_test_pred))
#print("Per-Class Accuracy:\n", classification_report(y_test, y_test_pred, digits=3))

# For training set
#df_train = pd.DataFrame({'true': y_train, 'pred': y_train_pred})
#accuracy_per_phase_train = df_train.groupby('true').apply(lambda x: accuracy_score(x['true'], x['pred']))

#print("Accuracy per phase (Train):")
#print(accuracy_per_phase_train)

# For test set
#df_test = pd.DataFrame({'true': y_test, 'pred': y_test_pred})
#accuracy_per_phase_test = df_test.groupby('true').apply(lambda x: accuracy_score(x['true'], x['pred']))

#print("\nAccuracy per phase (Test):")
#print(accuracy_per_phase_test)

## cross-validation:

param_grid = {
    'n_estimators': [500, 600],              
    'max_depth': [50, 60]    
    #'max_features': ['sqrt', 'log2']   
    #'min_samples_split': [2, 5],        
}

# Set up GridSearch with 10-fold cross-validation optimizing for accuracy
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=-1
)

# Fit the model to your data
grid_search.fit(X, y)

#Output best settings and best accuracy
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# kappa

#kappa_train = cohen_kappa_score(y_train, y_train_pred)
#kappa_test = cohen_kappa_score(y_test, y_test_pred)

#print(f"Cohen's Kappa (Train): {kappa_train:.3f}")
#print(f"Cohen's Kappa (Test): {kappa_test:.3f}")