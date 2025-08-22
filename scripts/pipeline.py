import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('data/data.csv')

# Data refinement (camada prata)
df.dropna(inplace=True)

# Features and target
X = df[['feature1', 'feature2']]
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Export results
with open('data/results.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy:.2f}\n')
