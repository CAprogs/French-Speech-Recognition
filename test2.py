import numpy as np
import pandas as pd
import librosa
import pickle
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from jiwer import wer

# Chemins vers les fichiers
path_data = '//Users//charles-albert//Desktop//Projet Ingénieur//treat_datas//'
path_models = '//Users//charles-albert//Desktop//Projet Ingénieur//Models//'

# Chargement des données
train_df = pd.read_csv(path_data + 'train.csv')
test_df = pd.read_csv(path_data + 'test.csv')

# Prétraitement audio (MFCC)
def extract_mfcc_features(audio_path, duration):
    audio, sr = librosa.load(audio_path, sr=None, duration=duration)
    mfcc = librosa.feature.mfcc(audio, sr=sr)
    return mfcc

train_df['mfcc_features'] = train_df.apply(lambda row: extract_mfcc_features(row['audio_path'], row['duration']), axis=1)
test_df['mfcc_features'] = test_df.apply(lambda row: extract_mfcc_features(row['audio_path'], row['duration']), axis=1)

# Tokenizer pour les séquences traduites
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['transcription'])

# Conversion des séquences en indices
train_sequences = tokenizer.texts_to_sequences(train_df['transcription'])
test_sequences = tokenizer.texts_to_sequences(test_df['transcription'])

# Paddage des séquences
max_sequence_length = max(max(len(seq) for seq in train_sequences), max(len(seq) for seq in test_sequences))
train_sequences_padded = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_sequences_padded = pad_sequences(test_sequences, maxlen=max_sequence_length)

# Paramètres pour le réseau de neurones
vocab_size = len(tokenizer.word_index) + 1
input_shape = train_df['mfcc_features'][0].shape
output_shape = max_sequence_length

# Création du modèle
def create_model():
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_shape, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Recherche par grille pour trouver le meilleur modèle
param_grid = {'batch_size': [16, 32, 64], 'epochs': [10, 20, 30]}
model = KerasClassifier(build_fn=create_model, verbose=1)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_search.fit(np.array(train_df['mfcc_features'].tolist()), train_sequences_padded)

# Meilleur modèle
best_model = grid_search.best_estimator_

# Évaluation du meilleur modèle sur les données de test
predictions = best_model.predict(np.array(test_df['mfcc_features'].tolist()))
wer_score = wer(test_sequences_padded, predictions)
print("WER (Word Error Rate):", wer_score)

# Sauvegarde du meilleur modèle
best_model.save(path_models + 'speech_to_text_model.h5')