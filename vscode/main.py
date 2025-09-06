from flask import Flask, jsonify, request
import joblib
import pandas as pd

from flask_cors import CORS   # <-- add this

app = Flask(__name__)
CORS(app)  # <-- enable CORS so HTML form can call API

# Load model and pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

@app.route("/diabetes", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert JSON to DataFrame
        df = pd.DataFrame([data])

        # Transform input using pipeline
        processed_data = pipeline.transform(df)

        # Predict
        prediction = model.predict(processed_data)[0]

        eligible = "No" if prediction == 1 else "Yes"
        eligibility_text = "Not Eligible to Donate" if prediction == 1 else "Eligible to Donate"

        return jsonify({
            "predicted_diabetes": int(prediction),
            "eligible_to_donate": eligible,
            "eligibility_text": eligibility_text
        })    
    except Exception as e:
        return jsonify({"Error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)