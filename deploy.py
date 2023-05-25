import pickle
import pandas as pd
import nltk
from joblib import load
from nltk.corpus import stopwords
from TweetsTransformer import TweetsTransformer
import numpy as np
from flask import Flask, request , render_template

preprocessor = TweetsTransformer(lower_eff=False,non_english_rm=False, meaningless_rm=False, arabic_norm=True)
nltk.download('stopwords')
arabic_stopwords = stopwords.words('arabic')

with open('vectorizer.pickle','rb') as file:
    tokenizer = pickle.load(file)

model = load('my_model.joblib')



from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict_text():
    input_text = request.form['input_text']
    prediction = predict(input_text)
    return render_template('index.html', input_text=input_text, prediction=prediction)

def predict(text):
    # Your code to generate a prediction based on the input text goes here
    l = ["مصري","لبناني","ليبي","مغربي","سعودي"]
    df = pd.DataFrame({'text': [text]})
    clean_text = preprocessor.transform(df['text'])
    tokenized_text = tokenizer.transform(clean_text)
    predictions = model.predict(tokenized_text)
    prediction = predictions[0]
    return l[prediction]

if __name__ == '__main__':
    app.run(debug=True)
