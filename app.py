from __future__ import division, print_function
from flask import Flask, render_template, url_for, request, redirect
from datetime import datetime
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from keras.models import load_model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize



app = Flask(__name__)
MODEL_PATH = 'C:\\Users\\DELL\\Desktop\\Sentiment Analysis\\final_model (1).h5'
model = load_model(MODEL_PATH)

def model_predict(text, model):
    ps = PorterStemmer()
    review=text.lower()
    nltk.download('stopwords')

    tokens = word_tokenize(review)
    words = [word for word in tokens if word.isalpha()]
    review = [ps.stem(word) for word in words if not word in stopwords.words('english')]
    review = ' '.join(review)
    voc_size=10000
    onehot_repr=[one_hot(review,voc_size)] 
    sent_length=20
    embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
    ans=model.predict([[embedded_docs]])
    print(ans)
    number=np.argmax(ans)
    number=int(number)
    if number==2:
        return "Positive"
    elif number==1:
        return "Neutral"
    else:
        return "Negative"
    
    

@app.route('/upload', methods=['POST','GET'])
def upload():
    if request.method == 'POST':
        text = request.form.get("t1")
        print(text)

        preds = model_predict(text, model)  
        return preds
    


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
    

    


















if __name__ == "__main__":
    app.run(debug=True)