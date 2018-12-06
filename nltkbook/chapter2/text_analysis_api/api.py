from flask import Flask
from flask import request, jsonify
from sklearn.externals import joblib


app = Flask(__name__)


VECTORIZER_MODEL_PATH = './text_analysis_vectorizer_1504879283.joblib'
CLASSIFIER_MODEL_PATH = './text_analysis_classifier_1504879283.joblib'

# Load the vectorizer and the classifier
vectorizer = joblib.load(VECTORIZER_MODEL_PATH)
classifier = joblib.load(CLASSIFIER_MODEL_PATH)


@app.route("/classify", methods=['POST'])
def classify():
    prediction = classifier.predict(
        vectorizer.transform([request.data]))[0]
    return jsonify(prediction=prediction)