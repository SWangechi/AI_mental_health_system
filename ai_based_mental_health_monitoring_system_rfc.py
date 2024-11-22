# -*- coding: utf-8 -*-
"""AI_Based_Mental_Health_Monitoring_System_rfc.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XdApHpRrMIX4sVilVSMM2JPlmLdmlwwK

# Sentiment Analysis for Social Media Data

## Import required libraries
"""

# pip install emoji

import pandas as pd
import numpy as np
import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer, PorterStemmer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup

"""## Download necessary NTLK resources"""

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

nltk.download('all')

"""## Step 2. Load Data from Different Social Media Sources"""

# Load Twitter data
twitter_data1 = pd.read_csv('/content/OCD_tweets.csv')
twitter_data1['platform'] = 'Twitter'
display(twitter_data1.head())

twitter_data2 = pd.read_csv('/content/UMHD__tweets.csv')
twitter_data2['platform'] = 'Twitter'
display(twitter_data2.head())

twitter_data3 = pd.read_csv('/content/MH_Campaigns_tweets1723.csv')
twitter_data3['platform'] = 'Twitter'
display(twitter_data3.head())

twitter_data4 = pd.read_csv('/content/MHAW__tweets.csv')
twitter_data4['platform'] = 'Twitter'
display(twitter_data4.head())

# Load Reddit data
reddit_data = pd.read_csv('/content/Combined_reddit_Data.csv')
reddit_data['platform'] = 'Reddit'
display(reddit_data.head())

# Load Forum data
forum_data = pd.read_csv('/content/mental_health_counseling_conversations.csv')
forum_data['platform'] = 'Forum'
display(forum_data.head())

"""## Step 3: Data Preprocessing

## Define Text Cleaning Function
This function will sequentially apply each of the specified preprocessing steps.
"""

# Define custom stopwords (add any additional words specific to mental health discourse if needed)
custom_stopwords = set(stopwords.words('english')).union({'specific', 'additional', 'filler', 'words'})

# Initialize spacy NLP model for NER removal
# nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    # Named Entity Removal (NER)
    # Remove Named Entities (NER) by removing capitalized words
    text = re.sub(r'\b[A-Z][a-z]*\b', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove numbers and written numbers
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\b', '', text)

    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)

    # Remove measurements (e.g., "mg," "cm," "kg")
    text = re.sub(r'\b(\d+|\d+\.\d+)(mg|g|cm|kg|m|%)\b', '', text)

    # Remove emojis
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Remove filler words and negations (customize list as needed)
    fillers = ['um', 'uh', 'like', 'so', 'you know', 'not']
    text = ' '.join([word for word in text.split() if word not in fillers])

    # Tokenization
    words = word_tokenize(text)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Stemming
    stemmer = SnowballStemmer('english')
    words = [stemmer.stem(word) for word in words]

    # Remove stopwords (including custom stopwords)
    words = [word for word in words if word not in custom_stopwords]

    return ' '.join(words)

"""## Apply Cleaning Function to Each Dataset"""

print(twitter_data1.columns)
print(twitter_data2.columns)
print(twitter_data3.columns)
print(twitter_data4.columns)
print(reddit_data.columns)
print(forum_data.columns)

# Apply cleaning on the 'tweet' columns in the Twitter datasets and save to 'cleaned_text'
twitter_data1['cleaned_text'] = twitter_data1['tweet'].astype(str).apply(clean_text)
twitter_data2['cleaned_text'] = twitter_data2['tweet'].astype(str).apply(clean_text)
twitter_data3['cleaned_text'] = twitter_data3['tweet'].astype(str).apply(clean_text)
twitter_data4['cleaned_text'] = twitter_data4['tweet'].astype(str).apply(clean_text)

# Apply cleaning on the 'statement' column in the Reddit dataset and save to 'cleaned_text'
reddit_data['cleaned_text'] = reddit_data['statement'].astype(str).apply(clean_text)

# Apply cleaning on the 'Context' column in the Forum dataset and save to 'cleaned_text'
forum_data['cleaned_text'] = forum_data['Context'].astype(str).apply(clean_text)

"""# Step 4:
## Merge Datasets and Remove Duplicates
"""

# Combine all datasets into one
merged_data = pd.concat([
    twitter_data1[['cleaned_text']],
    twitter_data2[['cleaned_text']],
    twitter_data3[['cleaned_text']],
    twitter_data4[['cleaned_text']],
    reddit_data[['cleaned_text']],
    forum_data[['cleaned_text']]
], ignore_index=True)

# Remove duplicates based on the 'cleaned_text' column
merged_data = merged_data.drop_duplicates(subset='cleaned_text')

# Save the merged and cleaned dataset to a CSV file
merged_data.to_csv('merged_cleaned_data.csv', index=False)

# Display a sample of the merged dataset
merged_data.head()

merged_data.head(30)

"""# Step 5: Anonymization
Anonymize Identifiable Information

The anonymize_text function masks usernames, links, email addresses, IP addresses, and some proper nouns to maintain privacy.

Creates a new column anonymized_text for anonymized content.
"""

import re

def anonymize_text(text):
    """
    Remove identifiable information such as usernames, locations, and email addresses.
    """
    text = re.sub(r'@\w+', '[USERNAME]', text)  # Mask usernames
    text = re.sub(r'http\S+|www\S+', '[LINK]', text)  # Mask links
    text = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', '[IP_ADDRESS]', text)  # Mask IP addresses
    text = re.sub(r'\b[A-Z][a-z]*\b', '[PLACE]', text)  # Mask proper nouns as locations (optional heuristic)
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)  # Mask email addresses
    return text

# Apply anonymization on the `cleaned_text` column
merged_data['anonymized_text'] = merged_data['cleaned_text'].apply(anonymize_text)

# Display a sample of anonymized data
print(merged_data[['cleaned_text', 'anonymized_text']].head())

"""# Step 7: Data Labelling

## Label Data with Known Mental Health Indicators
### Apply Mental Health Labels.

Define a dictionary of mental health-related keywords and patterns to label the data.

The assign_labels function uses regex to detect keywords and phrases indicative of mental health conditions.

Creates a new column with the following mental conditions labes:

**Bipolar Disorder** Keywords include bipolar, manic, and mania.

**PTSD:** Keywords include PTSD, trauma, flashbacks, and hypervigilance.

**Schizophrenia:** Keywords include schizophrenia, hallucinations, and delusions.

**Eating Disorder:** Keywords include eating disorder, anorexia, bulimia, and binge.

**ADHD:** Keywords include ADHD, attention deficit, hyperactive, and impulsivity.

**Self-Harm:** Keywords include self-harm, self injury, and cutting.

**Suicidal Ideation:** Keywords include suicide, suicidal, and hopelessness.

**Insomnia:** Keywords include insomnia, sleeplessness, and difficulty sleeping.

**Phobia:** Keywords include phobia, fear of, and irrational fear.
"""

import re

def assign_labels(text):
    """
    Label text based on the presence of specific mental health indicators.
    """
    labels = []
    if re.search(r'\b(OCD|obsessive|compulsive)\b', text, re.IGNORECASE):
        labels.append('OCD')
    if re.search(r'\b(anxiety|anxious|panic|fear)\b', text, re.IGNORECASE):
        labels.append('Anxiety')
    if re.search(r'\b(depression|depressed|hopeless|sadness)\b', text, re.IGNORECASE):
        labels.append('Depression')
    if re.search(r'\b(bipolar|manic|mania)\b', text, re.IGNORECASE):
        labels.append('Bipolar Disorder')
    if re.search(r'\b(PTSD|trauma|flashbacks|hypervigilance)\b', text, re.IGNORECASE):
        labels.append('PTSD')
    if re.search(r'\b(schizophrenia|hallucinations|delusions)\b', text, re.IGNORECASE):
        labels.append('Schizophrenia')
    if re.search(r'\b(eating disorder|anorexia|bulimia|binge)\b', text, re.IGNORECASE):
        labels.append('Eating Disorder')
    if re.search(r'\b(ADHD|attention deficit|hyperactive|impulsivity)\b', text, re.IGNORECASE):
        labels.append('ADHD')
    if re.search(r'\b(self-harm|self injury|cutting)\b', text, re.IGNORECASE):
        labels.append('Self-Harm')
    if re.search(r'\b(suicide|suicidal|hopelessness)\b', text, re.IGNORECASE):
        labels.append('Suicidal Ideation')
    if re.search(r'\b(insomnia|sleeplessness|difficulty sleeping)\b', text, re.IGNORECASE):
        labels.append('Insomnia')
    if re.search(r'\b(phobia|fear of|irrational fear)\b', text, re.IGNORECASE):
        labels.append('Phobia')
    return ', '.join(labels) if labels else 'None'

# Apply labels to the `anonymized_text` column
merged_data['mental_health_labels'] = merged_data['anonymized_text'].apply(assign_labels)

# Display a sample of data with labels
print(merged_data[['anonymized_text', 'mental_health_labels']].head())

merged_data.head(30)

from collections import Counter

# Flatten all labels into a single list
all_labels = merged_data['mental_health_labels'].str.split(', ').explode()

# Count the occurrences of each label
label_counts = Counter(all_labels)

# Convert to a DataFrame for better readability
label_counts_df = pd.DataFrame(label_counts.items(), columns=['Label', 'Count'])

# Sort by count in descending order
label_counts_df = label_counts_df.sort_values(by='Count', ascending=False).reset_index(drop=True)

# Display the count of mental health labels
print(label_counts_df)

"""## Further Data Cleaning"""

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters
    return text.lower()

merged_data['cleaned_text'] = merged_data['cleaned_text'].apply(clean_text)

"""### Tokenization and Lemmatization
Use NLTK or spaCy for text processing
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

merged_data['processed_text'] = merged_data['cleaned_text'].apply(preprocess_text)

"""### Exploratory Data Analysis (EDA)
Visualize label distributions
"""

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(y=merged_data['mental_health_labels'])
plt.title("Label Distribution")
plt.show()

merged_data.head()

# from sklearn.feature_extraction.text import CountVectorizer

# # Initialize CountVectorizer
# vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')

# # Fit and transform text data
# X = vectorizer.fit_transform(merged_data['processed_text']).toarray()

# # Display the shape of the Bag of Words matrix
# print("Shape of BoW:", X.shape)

# import pickle

# # Save the vectorizer
# with open('vectorizer.pkl', 'wb') as file:
#     pickle.dump(vectorizer, file)

"""### Train Baseline Models
Text Vectorization

Use TF-IDF for feature extraction
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

# Initialize TfidfVectorizer and classifier
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
classifier = SGDClassifier()

# Batch processing
batch_size = 10000
X_sparse_batches = []  # Store sparse matrices
y_batches = []         # Store corresponding labels

# Process data in batches
for i in range(0, len(merged_data), batch_size):
    batch_texts = merged_data['processed_text'][i:i + batch_size]
    batch_labels = merged_data['mental_health_labels'][i:i + batch_size]
    X_batch = tfidf.fit_transform(batch_texts)
    classifier.partial_fit(X_batch, batch_labels, classes=np.unique(merged_data['mental_health_labels']))
    X_sparse_batches.append(X_batch)
    y_batches.extend(batch_labels)

"""## Bag of Words"""

# Display the shape of the Bag of Words matrix
print("Shape of BoW:", X_batch.shape)

"""### Baseline Model Training
Train and evaluate classical models
"""

# Combine batches for train-test split
from scipy.sparse import vstack
X_full = vstack(X_sparse_batches)  # Stack sparse matrices into one
y_full = np.array(y_batches)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# Train RandomForestClassifier
rfc_model = RandomForestClassifier()
rfc_model.fit(X_train, y_train)

# Predictions and evaluation
predictions = rfc_model.predict(X_test)
print(classification_report(y_test, predictions))

"""### Fine-Tune BERT for Mental Health Signals
Load and Tokenize Data

Tokenize text using BERTTokenizer
"""

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer(list(merged_data['processed_text']), truncation=True, padding=True, max_length=128, return_tensors='pt')

"""### Model Fine-Tuning
Train a BERT-based classifier
"""

from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import test_torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# Encode string labels to integers using LabelEncoder
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(merged_data['mental_health_labels'])  # Replace with your labels column

# Define the custom dataset class
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: test_torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = test_torch.tensor(self.labels[idx], dtype=test_torch.long)  # Ensure labels are long type for classification
        return item

    def __len__(self):
        return len(self.labels)

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_encoder.classes_)  # Number of unique labels
)

# Tokenize the data
texts = list(merged_data['processed_text'])  # Replace 'processed_text' with your text column name

tokenized_data = tokenizer(
    texts,
    max_length=64,  # Reduce max_length for faster processing
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# Subset the data for quick debugging (comment this out for full dataset training)
subset_indices = list(range(500))  # Use first 500 samples for testing
tokenized_data = {key: val[subset_indices] for key, val in tokenized_data.items()}
labels = labels[subset_indices]

# Prepare the dataset
train_dataset = CustomDataset(tokenized_data, labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,  # Reduce the number of epochs
    per_device_train_batch_size=32,  # Increase batch size
    evaluation_strategy='epoch',
    save_steps=5000,  # Adjust save steps to reduce overhead
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=1000,  # Reduce logging frequency
    report_to=None,  # Disable logging to W&B for faster training
    fp16=True,  # Enable mixed precision training
)

# Create the Trainer
trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset  # For demonstration; use a proper validation set in practice
)

# Train the model
trainer.train()

from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments, TrainerCallback
import test_torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# Encode string labels to integers using LabelEncoder
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(merged_data['mental_health_labels'])  # Replace with your labels column

# Define the custom dataset class
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: test_torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = test_torch.tensor(self.labels[idx], dtype=test_torch.long)  # Ensure labels are long type for classification
        return item

    def __len__(self):
        return len(self.labels)

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_encoder.classes_)  # Number of unique labels
)

# Tokenize the data
texts = list(merged_data['processed_text'])  # Replace 'processed_text' with your text column name

tokenized_data = tokenizer(
    texts,
    max_length=64,  # Reduce max_length for faster processing
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# Subset the data for quick debugging (comment this out for full dataset training)
subset_indices = list(range(500))  # Use first 500 samples for testing
tokenized_data = {key: val[subset_indices] for key, val in tokenized_data.items()}
labels = labels[subset_indices]

# Prepare the dataset
train_dataset = CustomDataset(tokenized_data, labels)

# Define custom logging callback
class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"Logs at step {state.global_step}: {logs}")

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,  # Reduce the number of epochs
    per_device_train_batch_size=32,  # Increase batch size
    evaluation_strategy='epoch',
    save_steps=5000,  # Adjust save steps to reduce overhead
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10,  # Log every 10 steps
    disable_tqdm=False,  # Ensure progress bar is displayed
    report_to=None,  # Disable logging to W&B for faster training
    fp16=True,  # Enable mixed precision training
)

# Create the Trainer
trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,  # For demonstration; use a proper validation set in practice
    callbacks=[CustomLoggingCallback()]  # Add the custom logging callback
)

# Train the model
trainer.train()

"""## Evaluate the BERT Model"""

from sklearn.metrics import accuracy_score

# Evaluate the model
eval_results = trainer.evaluate()

# Print evaluation results
print(f"Evaluation Results: {eval_results}")

"""## Save the Models

### Saving the BERT Model
"""

# Save the trained BERT model and tokenizer
model_save_path = "./saved_models/bert_model"
tokenizer_save_path = "./saved_models/bert_tokenizer"

# Save the model
bert_model.save_pretrained(model_save_path)

# Save the tokenizer
tokenizer.save_pretrained(tokenizer_save_path)

print(f"BERT model and tokenizer saved to {model_save_path} and {tokenizer_save_path}.")

"""### Saving the Random Forest Model"""

import joblib

# Save the RandomForest model
rfc_model_save_path = "./saved_models/random_forest_model.pkl"

joblib.dump(rfc_model, rfc_model_save_path)

print(f"RandomForest model saved to {rfc_model_save_path}.")

# !zip -r logs.zip logs/

# !zip -r saved_models.zip saved_models/

# !zip -r results.zip results/

# !zip -r wandb.zip wandb/

"""# Build Real-Time Monitoring Pipeline

## Real-Time Data Collection

Use APIs to fetch data (e.g., Twitter)
"""

# pip install tweepy

# import tweepy

# client = tweepy.Client(bearer_token="YOUR_BEARER_TOKEN")
# tweets = client.search_recent_tweets(query="mental health", max_results=100)

"""## Prediction Pipeline
Process new text data in real-time using your trained model:

"""

# def predict_mental_health(text, model, tokenizer):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
#     outputs = model(**inputs)
#     return outputs.logits.argmax(dim=1).item()

"""## Create a Dashboard
Visualizations

Use Python tools like Dash or Streamlit for real-time charts:
"""

# pip install streamlit

# import streamlit as st

# st.title("Mental Health Monitoring Dashboard")
# st.line_chart(sentiment_trends)  # Replace with real-time data

"""## Personalized Alerts
Trigger notifications based on thresholds
"""

# if sentiment_score < threshold:
#     st.warning("Alert: Potential mental health issue detected!")