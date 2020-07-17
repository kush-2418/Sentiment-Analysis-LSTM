from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model


IMAGE_FOLDER = os.path.join('static', 'img_pool')
word_to_id = imdb.get_word_index()
sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)
model = load_model('sentiment_analysis_lstm.h5')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

    
@app.route('/', methods=['GET', 'POST'])
def home():

    return render_template("home.html")


@app.route('/sentiment_analysis_prediction',methods=["POST","GET"])
def sentiment_analysis_prediction():
    if request.method=='POST':


        text = request.form['text']
        sentiment = ''
        max_review_length = 500
        
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        text = text.lower().replace("<br />", " ")
        text=re.sub(strip_special_chars, "", text.lower())

        words = text.split() #split string into a list
        x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word]<=20000) else 0 for word in words]]
        x_test = sequence.pad_sequences(x_test, maxlen=max_review_length) # Should be same which you used for training data
        vector = np.array([x_test.flatten()])
        with graph.as_default():
            set_session(sess)
            probability = model.predict(array([vector][0]))[0][0]
            class1 = model.predict_classes(array([vector][0]))[0][0]
        if class1 == 0:
            sentiment = 'Negative'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sad_emoji.png')
        else:
            sentiment = 'Positive'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'happy_emoji.png')
    return render_template('home.html', text=text, sentiment=sentiment, probability=probability, image=img_filename)

if __name__ == "__main__":
    app.run()
