#!iApp/views.py
from django.shortcuts import render
from keras.models import model_from_json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from classifier import SentimentClassifier


def process_news(text, type_):
    MAX_NB_WORDS = None
    MAX_SEQUENCE_LENGTH = None
    if type_ == "CNN":
        MAX_SEQUENCE_LENGTH = 450
        MAX_SEQUENCE_LENGTH = 15
    else:
        MAX_SEQUENCE_LENGTH = 350
        MAX_SEQUENCE_LENGTH = 20
    d = {'texto': [text]}
    df = pd.DataFrame(data=d)
    corpus = df["texto"]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(corpus.values)
    word_index = tokenizer.word_index
    X = tokenizer.texts_to_sequences(corpus.values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    return X


def clasificador(request):
    """View para clasificar un texto"""
    if request.method == 'POST':
        name = None
        if request.POST["type"] == "CNN" or request.POST["type"] == "SVM":
            name = "model"
        else:
            name = "modelLSTM"
        json_file = open('iApp/'+ name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("iApp/" + name + ".h5")
        labels = ['crime', 'economi', 'educ', 'health', 'polit', 'sport']
        news = request.POST["data"]
        sentiment_pred = sentiment(news)
        print(sentiment_pred)
        ynew = loaded_model.predict(process_news(news, request.POST["type"]))
        K.clear_session()
        index_max = np.argmax(ynew[0])
        print("result: " + labels[index_max])
        data = {"data": news, "result": labels[index_max] + " - " + sentiment_pred}
        return render(request, "clasificador.html", data)
    return render(request, "clasificador.html",)

def sentiment(news):
    clf = SentimentClassifier()
    if clf.predict(news) >= 0.5:
        return "positive"
    return "negative"
