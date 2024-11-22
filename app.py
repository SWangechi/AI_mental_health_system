from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import joblib

# Load Models
bert_model = BertForSequenceClassification.from_pretrained("./saved_models/bert_model").to("cpu")
tokenizer = BertTokenizer.from_pretrained("./saved_models/bert_tokenizer")
rfc_model = joblib.load("./saved_models/random_forest_model.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    model_type = data.get('model', 'bert')  # Default to BERT

    if model_type == 'bert':
        # Preprocess input for BERT
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = bert_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        return jsonify({'model': 'BERT', 'prediction': prediction})
    elif model_type == 'random_forest':
        # Preprocess input for Random Forest
        vectorized_input = tokenizer.encode(text, return_tensors="pt")
        prediction = rfc_model.predict(vectorized_input)
        return jsonify({'model': 'RandomForest', 'prediction': prediction[0]})
    else:
        return jsonify({'error': 'Invalid model specified.'}), 400

if __name__ == '__main__':
    app.run(debug=True)