from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
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
model = load_model("model/indonesian_general_election_sgd_tfidf.pkl")
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/sentiment', methods=['POST'])
def sentiment():
    # return render_template('sentiment.html')
    text = request.form['text']
    data = [text]

    pred = model.predict(data)
    return render_template('sentiment.html', prediction=pred)
