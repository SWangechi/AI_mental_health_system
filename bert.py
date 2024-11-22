from transformers import BertTokenizer, BertForSequenceClassification

model_save_path = "./saved_models/bert_model"
tokenizer_save_path = "./saved_models/bert_tokenizer"

# Load model and tokenizer
bert_model = BertForSequenceClassification.from_pretrained(model_save_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_save_path)

print("BERT model and tokenizer loaded successfully.")
