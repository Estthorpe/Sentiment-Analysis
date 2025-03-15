from flask import Flask, request, jsonify
import joblib

#Load the trained model
model = joblib.load("naive_bayes_sentiment_model.pkl")

#initialize Flask app
app = Flask(__name__)

#Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "text" not in data:
        return jsonify({"error": "Missing 'text' parameter"}), 400
    
    text = data["text"].lower()  # Convert text to lowercase
    prediction = model.predict([text])[0]  # Predict sentiment

    sentiment_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
    return jsonify({"sentiment": sentiment_map[prediction]})

# ✅ Run the Flask app (for local deployment)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)