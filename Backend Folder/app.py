from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

# Load the trained model and vectorizer
with open("model/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    news_text = data.get("news", "").strip()

    if not news_text:
        return jsonify({"error": "No news text provided"}), 400

    # Transform input text using the pre-trained vectorizer
    text_vector = vectorizer.transform([news_text])  # No .fit_transform()

    # Make Prediction
    prediction = model.predict(text_vector)[0]
    result = "fake" if prediction == 1 else "real"

    return jsonify({"prediction": result})

# Run the Flask App
if __name__ == "__main__":
    app.run(debug=True)
