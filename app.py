from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained pipeline
with open("preprocess_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Define the home route
@app.route("/")
def home():
    return render_template("index.html")

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    # Extract input data from the form
    sepal_length = float(request.form.get("sepal_length"))
    sepal_width = float(request.form.get("sepal_width"))
    petal_length = float(request.form.get("petal_length"))
    petal_width = float(request.form.get("petal_width"))
    
    # Create input array
    inputs = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make prediction
    prediction = pipeline.predict(inputs)
    classes = ["Setosa", "Versicolor", "Virginica"]  # Class names
    
    # Return the result
    return render_template(
        "index.html",
        prediction_text=f"The predicted Iris species is: {classes[prediction[0]]}",
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)
