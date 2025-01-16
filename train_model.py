import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# Define preprocessing and model pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Standardize features
    ("classifier", RandomForestClassifier())  # RandomForestClassifier
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Save pipeline
with open("preprocess_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)
print("Pipeline trained and saved as preprocess_pipeline.pkl")
