import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
df = pd.read_csv("Data/top40_cell_cycle.csv")

# Combine 'M' and 'G2' into one class, if desired (optional)
df['phase'] = df['phase'].replace({'M': 'G2'})

# Define features and target
X = df.drop(columns=['phase', 'age'])
y = df['phase']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=949, stratify=y)

# Fit multinomial logistic regression with regularization
logreg = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',  # good for multinomial
    max_iter=1000,
    C=1.0,  # inverse of regularization strength
    random_state=949
)
logreg.fit(X_train, y_train)

# Predict
y_train_pred = logreg.predict(X_train)
y_test_pred = logreg.predict(X_test)

# Evaluation
# Output results
print("=== Training Set ===")
print("Overall Accuracy:", accuracy_score(y_train, y_train_pred))

df_train = pd.DataFrame({'true': y_train, 'pred': y_train_pred})
accuracy_per_phase_train = df_train.groupby('true').apply(lambda x: accuracy_score(x['true'], x['pred']))

print("Accuracy per phase (Train):")
print(accuracy_per_phase_train)

print("\n=== Test Set ===")
print("Overall Accuracy:", accuracy_score(y_test, y_test_pred))

df_test = pd.DataFrame({'true': y_test, 'pred': y_test_pred})
accuracy_per_phase_test = df_test.groupby('true').apply(lambda x: accuracy_score(x['true'], x['pred']))

print("\nAccuracy per phase (Test):")
print(accuracy_per_phase_test)

print("\n=== Confusion Matrix (Test Set) ===")
print(confusion_matrix(y_test, y_test_pred))
