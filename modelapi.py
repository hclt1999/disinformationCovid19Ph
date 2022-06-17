import os
import pandas as pd
import itertools
import collections
import re
import numpy as np
import ast
import nltk

from keras.layers import Dropout, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import regularizers
from sklearn import metrics
import pickle
import warnings
import logging
import transformers
from transformers import RobertaTokenizer, TFRobertaModel
import tensorflow as tf

from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import json

import nltk
nltk.download('stopwords')
#English Stopwords
from nltk.corpus import stopwords

#Import stopwords
stop = stopwords.words('english')
stop.sort()
fil_stop = ["akin","aking","ako","alin","am","amin","aming","ang","ano","anumang","apat","at","atin","ating","ay","bababa","bago","bakit","bawat","bilang","dahil","dalawa","dapat","din","dito","doon","gagawin","gayunman","ginagawa","ginawa","ginawang","gumawa","gusto","habang","hanggang","hindi","huwag","iba","ibaba","ibabaw","ibig","ikaw","ilagay","ilalim","ilan","inyong","isa","isang","itaas","ito","iyo","iyon","iyong","ka","kahit","kailangan","kailanman","kami","kanila","kanilang","kanino","kanya","kanyang","kapag","kapwa","karamihan","katiyakan","katulad","kaya","kaysa","ko","kong","kulang","kumuha","kung","laban","lahat","lamang","likod","lima","maaari","maaaring","maging","mahusay","makita","marami","marapat","masyado","may","mayroon","mga","minsan","mismo","mula","muli","na","nabanggit","naging","nagkaroon","nais","nakita","namin","napaka","narito","nasaan","ng","ngayon","ni","nila","nilang","nito","niya","niyang","noon","o","pa","paano","pababa","paggawa","pagitan","pagkakaroon","pagkatapos","palabas","pamamagitan","panahon","pangalawa","para","paraan","pareho","pataas","pero","pumunta","pumupunta","sa","saan","sabi","sabihin","sarili","sila","sino","siya","tatlo","tayo","tulad","tungkol","una","walang",'in']
collection_words = ['covid19ph', 'covid19', 'bakuna', 'resbakuna', 'coronavirus', '#covid19']
other_fil_stopwords = ['in','yung','higit','nang','wala','di','po', 'ba', 'ah', 'lang', 'yan', 'yang','walang', 'kayo', 'niyong', 'rin', 'mo', 'diyan', 'jan', 'nyo', 'e']

ROBERTA_DISINFORMATION = './roberta-fil-base.h5'

MAX_LEN = 280
MODEL_NAME = "jcblaise/roberta-tagalog-base"

def roberta_encode(texts, tokenizer):
    ct = len(texts)
    input_ids = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32') # Not used in text classification

    for k, text in enumerate(texts):
        # Tokenize
        tok_text = tokenizer.tokenize(text)
        
        # Truncate and convert tokens to numerical IDs
        enc_text = tokenizer.convert_tokens_to_ids(tok_text[:(MAX_LEN-2)])
        
        input_length = len(enc_text) + 2
        input_length = input_length if input_length < MAX_LEN else MAX_LEN
        
        # Add tokens [CLS] and [SEP] at the beginning and the end
        input_ids[k,:input_length] = np.asarray([0] + enc_text + [2], dtype='int32')
        
        # Set to 1s in the attention input
        attention_mask[k,:input_length] = 1

    return {
        'input_word_ids': input_ids,
        'input_mask': attention_mask,
        'input_type_ids': token_type_ids
    }

def build_model(n_categories):
    #with tpu_strategy.scope():
  input_word_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_word_ids')
  input_mask = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_mask')
  input_type_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_type_ids')

  # Import RoBERTa model from HuggingFace
  roberta_model = TFRobertaModel.from_pretrained(MODEL_NAME)
  x = roberta_model(input_word_ids, attention_mask=input_mask, token_type_ids=input_type_ids)

  # Huggingface transformers have multiple outputs, embeddings are the first one,
  # so let's slice out the first position
  x = x[0]

  x = tf.keras.layers.Dropout(0.1)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(256, activation='relu')(x)
  x = tf.keras.layers.Dense(n_categories, activation='softmax')(x)

  model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=x)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(lr=1e-5),
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])

  return model


def model_preprocess(input):
  if not input:
    raise Exception('Input is empty')
  df = pd.DataFrame(columns=['tweet'])
  df = df.append({'tweet':input}, ignore_index=True)

  ###Preprocessing
  #Lowercase the text
  df.tweet = df['tweet'].apply(lambda x: x.lower())
  #Remove stopwords
  df.tweet = df['tweet'].apply(lambda x: ' '.join([word for word in x.split(" ") if word not in (stop)]))
  #Remove Filipino stopwords
  df.tweet = df['tweet'].apply(lambda x: ' '.join([word for word in x.split(" ") if word not in (fil_stop)]))
  #Remove special characters
  df.tweet = df['tweet'].apply(lambda x: " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", x).split()))
  #Remove collection words
  df.tweet = df['tweet'].apply(lambda x: ' '.join([word for word in x.split(" ") if word not in (collection_words)]))
  #Remove excess Filipino stopwords
  df.tweet = df['tweet'].apply(lambda x: ' '.join([word for word in x.split(" ") if word not in (other_fil_stopwords)]))
  if not df.tweet[0]:
    raise Exception('String is blank because it only contained stopwords')
  return df.tweet[0]

app = Flask(__name__)
api = Api(app)

# Create parser for the payload data
parser = reqparse.RequestParser()
parser.add_argument('data')

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME) #Tokenizer
model = build_model(4)
model.load_weights(ROBERTA_DISINFORMATION)

# Define how the api will respond to the post requests
class DisinformationClassifier(Resource):
  def post(self):
    args = parser.parse_args()
    input = args['data']
    preprocess_output = model_preprocess(input)
    model_input = roberta_encode([preprocess_output], tokenizer)
    predicted = model.predict(model_input)
    y_prediction = np.argmax (predicted, axis = 1)
    output_map = {0: 'False', 1: 'Mostly False', 2: 'Mostly True', 3: 'True'}
    return output_map[y_prediction[0]]

api.add_resource(DisinformationClassifier, '/dc')

if __name__ == '__main__':
  app.run(port=8181)
