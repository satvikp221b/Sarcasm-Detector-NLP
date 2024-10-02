from flask import Flask, render_template, request
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import string
from textblob import TextBlob
import random
import json

app = Flask(__name__)


logistic_model = joblib.load('models/best_logistic_regression_model.pkl')
logistic_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

lstm_model = load_model('models/best_lstm_model.keras')
lstm_tokenizer = joblib.load('models/tokenizer.pkl')  


bert_model = TFBertForSequenceClassification.from_pretrained('best_bert_model')
bert_tokenizer = BertTokenizer.from_pretrained('bert_tokenizer')

lasso_model = joblib.load('models/best_logistic_regression_gridsearch_l1.pkl')
scaler = joblib.load('models/scaler.pkl')


MAX_LEN=128
# Define a function to display model information
def get_model_info(model_type):
    if model_type == 'logistic':
        # Logistic Regression Hyperparameters
        return {
            'Model': 'Logistic Regression',
            'Regularization Strength (C)': logistic_model.C if hasattr(logistic_model, 'C') else 'Default',
            'Solver': logistic_model.solver if hasattr(logistic_model, 'solver') else 'Default',
            'Max Iterations': logistic_model.max_iter if hasattr(logistic_model, 'max_iter') else 'Default'
        }
    elif model_type == 'lstm':
        # LSTM Hyperparameters
        optimizer_config = lstm_model.optimizer.get_config()
        return {
            'Model': 'LSTM',
            'Layers': len(lstm_model.layers),
            'Optimizer': optimizer_config['name'],
            'Loss Function': lstm_model.loss,
            'Epochs Trained': 5,
            'RandomTuner':'5 executions, 5 epoch, 2 trials each'
        }
    elif model_type == 'bert':
        return {
            'Model': 'BERT (TFBertForSequenceClassification)',
            'Max Sequence Length': MAX_LEN,
            'Number of Classes': 2,
            'Tokenizer': 'BertTokenizer',
        }
    elif model_type == 'logistic_l1':
        # Logistic Regression Hyperparameters
        return {
            'Model': 'Logistic Regression Lasso (Additional Features)',
            'Regularization Strength (C)': lasso_model.C if hasattr(logistic_model, 'C') else 'Default',
            'Solver': lasso_model.solver if hasattr(logistic_model, 'solver') else 'Default',
            'Max Iterations': lasso_model.max_iter if hasattr(logistic_model, 'max_iter') else 'Default',
            'Additional Features':'sentence_length,num_punctuations,num_exclamations,polarity_flip,Type_label,positive_words_count,positive_hyperbole_interaction,negative,negative_rhetorical_interaction,polarity_shift_label'
        }

def predict_logistic(text):
    # Transform text using the logistic vectorizer
    text_transformed = logistic_vectorizer.transform([text])
    prediction = logistic_model.predict(text_transformed)
    return 'Sarcastic' if prediction == 1 else 'Not Sarcastic'

def predict_lstm(text):
    # Tokenize and pad the input text for LSTM model
    text_tokenized = lstm_tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(text_tokenized, maxlen=100)  # Ensure the length matches the training
    prediction = lstm_model.predict(np.array(text_padded))[0][0]
    return 'Sarcastic' if prediction > 0.5 else 'Not Sarcastic'

def predict_bert(text):
    inputs = bert_tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=MAX_LEN)
    outputs = bert_model(inputs['input_ids'])
    prediction = tf.nn.softmax(outputs.logits, axis=1)
    predicted_class = tf.argmax(prediction, axis=1).numpy()[0]    
    return "Sarcastic" if predicted_class == 1 else "Not Sarcastic"    

def polarity_shift(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 0
    elif polarity < 0:
        return 2
    else:
        return 1
    
def calculate_additional_features(text,text_type):
    # Example: number of words, number of punctuation marks, etc.
    num_words = len(text.split())
    
    num_punctuation = sum([1 for char in text if char in string.punctuation])
    
    num_exclamation= text.count('!')
    text_type_dict={'GEN':0,'HYP':1,'RQ':2}
    blob = TextBlob(text)
    sentiment_list = [sentence.sentiment.polarity for sentence in blob.sentences]
    polarity_flip = any((x > 0 and y < 0) or (x < 0 and y > 0) for x, y in zip(sentiment_list[:-1], sentiment_list[1:]))
    polarity_flip=1 if polarity_flip else 0
    
    i=random.randint(1,10000)
    type_label=text_type_dict[text_type]

    positive_words = sum([1 for word in blob.words if TextBlob(word).sentiment.polarity > 0])

    positive_hyperbole_interaction= positive_words if text_type == 'HYP' else 0
    
    negative_words= sum([1 for word in blob.words if TextBlob(word).sentiment.polarity < 0])
    
    negative_rhetorical_interaction = negative_words if text_type=='RQ' else 0
    
    polarity = polarity_shift(text)

    additional_features = np.array([i,num_words, num_punctuation,num_exclamation,polarity_flip,type_label,positive_words,positive_hyperbole_interaction,negative_words,negative_rhetorical_interaction,polarity])

    return additional_features

def predict_lasso(text,text_type):
    # Step 1: Vectorize the input text using the loaded TF-IDF vectorizer
    text_tfidf = logistic_vectorizer.transform([text])  

    # Step 2: Calculate additional features from the input text
    additional_features = calculate_additional_features(text,text_type)
        
    # Step 3: Scale additional features (if scaling was applied during training)
    additional_features_scaled = scaler.transform([additional_features])

    # Step 4: Combine the TF-IDF text features and additional features
    # Convert text_tfidf to dense to concatenate with additional features
    combined_features = np.hstack((text_tfidf.toarray(), additional_features_scaled))

    # Step 5: Use the model to make a prediction
    prediction = lasso_model.predict(combined_features)
    
    # Convert the prediction to human-readable output
    return "Sarcastic" if prediction[0] == 1 else "Non-Sarcastic"  

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    selected_model = None
    model_info = None
    input_text = None
    model_performance=None
    model_dict={'logistic':'logistic_regression_metrics','lstm':'lstm_metrics','bert':'bert_metrics','logistic_l1':'logistic_regression_feature_metrics'}
    if request.method == 'POST':
        input_text = request.form['text']
        selected_model = request.form['model']

        if selected_model == 'logistic':
            result = predict_logistic(input_text)
        elif selected_model == 'lstm':
            result = predict_lstm(input_text)
        elif selected_model=='bert':
            result= predict_bert(input_text)
        elif selected_model == 'logistic_l1':
            text_type = request.form['text_type']  # Get the selected type of text
            result = predict_lasso(input_text, text_type)
        
        model_info = get_model_info(selected_model)
        with open(f"metrics/{model_dict[selected_model]}.json") as f:
            model_performance = json.load(f)
    
    return render_template('index.html', result=result, text=input_text, model=selected_model, model_info=model_info,model_performance=model_performance)

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
