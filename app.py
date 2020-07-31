from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from datetime import datetime
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from jcopml.utils import load_model

app = Flask(__name__)
# model = load_model("model/indonesian_general_election_sgd_tfidf.pkl")
bootstrap = Bootstrap(app)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/sentiment', methods=['POST'])
def sentiment():
    # # MODEL USING PIPELINE -> not successful on deployment
    # text = request.form['text']
    # data = [text]

    # pred = model.predict(data)
    # return render_template('sentiment.html', prediction=pred)

    # NOT USING PIPELINE
    df = pd.read_csv("data/preprocessed_df.csv", encoding="ISO-8859-1")
    df_data = df[["Isi_Tweet", "Sentimen"]]

    # dataset splitting
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    X = df_data["Isi_Tweet"]
    y = df_data["Sentimen"]

    cv = TfidfVectorizer()
    X = cv.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # training
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(tol=0.0003, loss='modified_huber',
                        penalty='l2', alpha=0.0002, random_state=42, max_iter=5)
    clf.fit(X_train, y_train)

    # return render_template('sentiment.html')
    if request.method == "POST":
        text = request.form['text']
        data = [text]
        data = cv.transform(data).toarray()
        pred = clf.predict(data)
    return render_template('sentiment.html', prediction=pred)
    # ******************************************************************
