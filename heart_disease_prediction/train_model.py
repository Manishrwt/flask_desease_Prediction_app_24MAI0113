# train_model.py
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate a synthetic dataset (substitute with actual heart disease dataset if available)
X, y = make_classification(
    n_samples=1000, 
    n_features=3, 
    n_informative=2,    # Number of informative features
    n_redundant=0,      # Number of redundant features
    n_classes=2, 
    random_state=42
)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")
