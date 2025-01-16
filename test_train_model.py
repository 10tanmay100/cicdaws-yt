import unittest
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Import the training function (if encapsulated in a script)
from train_model import pipeline,X_train, X_test, y_train, y_test

class TestIrisModel(unittest.TestCase):

    def setUp(self):
        """Set up for the tests"""
        # Ensure the artifacts folder exists
        self.artifacts_path = "artifacts"
        os.makedirs(self.artifacts_path, exist_ok=True)
        self.model_file_path = os.path.join(self.artifacts_path, "preprocess_pipeline.pkl")

    def test_model_training(self):
        """Test if the model trains without errors"""
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        train_accuracy = clf.score(X_train, y_train)
        self.assertGreater(train_accuracy, 0.8, "Training accuracy is too low!")

    def test_model_saving(self):
        """Test if the trained model is saved correctly"""
        # Save the model
        with open(self.model_file_path, "wb") as f:
            pickle.dump(pipeline, f)

        # Check if the file exists
        self.assertTrue(os.path.exists(self.model_file_path), "Model file was not saved!")

    def test_model_loading(self):
        """Test if the saved model can be loaded and used"""
        # Save and load the model
        with open(self.model_file_path, "wb") as f:
            pickle.dump(pipeline, f)

        with open(self.model_file_path, "rb") as f:
            loaded_model = pickle.load(f)

        # Ensure the loaded model predicts correctly
        predictions = loaded_model.predict(X_test)
        self.assertEqual(len(predictions), len(y_test), "Prediction length mismatch!")

    def test_model_prediction(self):
        """Test if the model can predict on new data"""
        sample_input = X_test[:1]  # Use a single sample
        prediction = pipeline.predict(sample_input)
        self.assertIn(prediction[0], [0, 1, 2], "Prediction is out of expected range!")

    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.model_file_path):
            os.remove(self.model_file_path)

if __name__ == "__main__":
    unittest.main()
