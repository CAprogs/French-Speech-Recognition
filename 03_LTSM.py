print('Importation des librairies...\n')
import numpy as np
import pandas as pd
import librosa
import pickle
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from jiwer import wer
#from keras.layers import Dense, Embedding, LSTM
#from keras.models import Sequential
#from keras.callbacks import EarlyStopping, ModelCheckpoint
#from sklearn.preprocessing import MultiLabelBinarizer


# Chemins vers les fichiers
path_1 = '//Users//charles-albert//Desktop//Projet Ingénieur//treat_datas//'
path_2 = '//Users//charles-albert//Desktop//Projet Ingénieur//Models//'
path_3 = '//Users//charles-albert//Desktop//Projet Ingénieur//datas//fr//'

print('Chargement des Datas...')
train_df = pd.read_csv(f'{path_1}train.tsv', sep='\t')                                  # path ( traité ) et sentence ( non traité )
test_df = pd.read_csv(f'{path_1}test.tsv', sep='\t')
durations_df = pd.read_csv(f'{path_3}clip_durations.tsv', sep='\t')

#################################################### TEST sur un petit ensemble
# Sélectionner n éléments aléatoires sans modifier le DataFrame d'origine
sample_train_df = train_df.sample(n=280, replace=False) 
sample_test_df = test_df.sample(n=120, replace=False)
####################################################

# Afficher le nouveau DataFrame
print('\nSample training :',sample_train_df.head(5))                                                        # train_df
print('\nSample test :',sample_test_df.head(5))                                                               # test_df

######################################################################################################## Traitement séquences
"""
# Remplacement des caratères spéciaux $,€ et %
def replace(Dataframe):
    char = {'$':'dollars', '€':'euros', '%':'pourcent'}
    for sentence in Dataframe['sentence']:
        for i in char:
            if str(' '+i) in sentence:
                Dataframe['sentence'] = Dataframe['sentence'].str.replace(i, str(char[i]))
            elif i in sentence:
                Dataframe['sentence'] = Dataframe['sentence'].str.replace(i, str(' '+char[i]))
            else:
                continue
    return Dataframe
"""
def replace(Dataframe):
    char = {'$':'dollars', '€':'euros', '%':'pourcent'}
    for index, sentence in enumerate(Dataframe['sentence']):
        for i in char:
            if str(' '+i) in sentence:
                Dataframe.loc[index, 'sentence'] = Dataframe['sentence'].str.replace(i, str(char[i]))
            elif i in sentence:
                Dataframe.loc[index, 'sentence'] = Dataframe['sentence'].str.replace(i, str(' '+char[i]))
            else:
                continue
    return Dataframe

print('\nTraitement des caractères spéciaux ($,€ et %)...\n')
sample_train_df = replace(sample_train_df) # train_df
sample_test_df = replace(sample_test_df) # test_df

print('Creation du Tokenizer...\n')
# Créer un tokenizer
tokenizer = Tokenizer(filters='!"#&()*+,-./:;<=>?@[\\]^_`{|}~',lower=True,split= " ",oov_token="<unk>") 
tokenizer.fit_on_texts(sample_train_df['sentence']) # train_df

# Convertir le texte en séquences numériques
train_sequences = tokenizer.texts_to_sequences(sample_train_df['sentence'])                                         # train_df
test_sequences = tokenizer.texts_to_sequences(sample_test_df['sentence'])                                           # test_df

# Obtenir la taille du vocabulaire
vocab_size = len(tokenizer.word_index) + 1
print('Taille du vocabulaire :', vocab_size)

# Paddage des séquences pour qu'elles aient la même longueur
max_sequence_length = 100  # Définir la longueur maximale souhaitée
train_sequences_padded = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_sequences_padded = pad_sequences(test_sequences, maxlen=max_sequence_length)

# Ajouter les séquences paddées dans une nouvelle colonne 'sequences'
sample_train_df['sequences'] = train_sequences_padded.tolist()                                                       # train_df
sample_test_df['sequences'] = test_sequences_padded.tolist()                                                         # test_df

print('\nEnregistrement du Tokenizer...\n')
with open(f'{path_2}tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

###################################################################################################### Fin Traitement séquences

###################################################################################################### Traitement audio
print('\nTraitement de l\'audio...')

# Resampler les audios
def preprocess_audio(path_audio, target_sr):
    signal, sr = librosa.load(path_audio, sr=None)  # Charger l'audio avec le sampling rate d'origine
    if sr != target_sr:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)  # Resampler l'audio au sampling rate cible
    return np.asarray(signal) # retourne une matrice numpy

def treat_audio(path_audio,max_duration):
    target_sr = 32000 # Sampling rate cible
    segment_duration = 1

    # Resampler le fichier audio au target_sr 
    audio_resample = preprocess_audio(path_audio,target_sr)

    # Normalisation
    y_norm = librosa.util.normalize(audio_resample)

    # Ajouter des zéros à la fin du fichier audio pour qu'il ait la même durée que les autres
    audio_resample = librosa.util.pad_center(y_norm, size=max_duration * target_sr)

    # Découper les fichiers audio en segments de taille fixe
    segment = librosa.util.frame(audio_resample, frame_length=int(segment_duration),hop_length=int(segment_duration) )

    return segment

# Trouver la durée maximale des deux fichiers audio
max_duration = int(durations_df['duration'].max()/1000)  # extraire la durée max de l'audio
print('\nDurée maximale d\'un audio : ',max_duration)

# Ajout de colonnes 'processed_audio' à chaque dataframe
sample_train_df['processed_audio'] = sample_train_df['path'].apply(lambda path: treat_audio(path, max_duration))                    # train_df 
sample_test_df['processed_audio'] = sample_test_df['path'].apply(lambda path: treat_audio(path, max_duration))                      # test_df

###################################################################################################### Fin Traitement audio

# Definir les entrées et sorties du modèle
            # Entrées
X_train = np.array(sample_train_df['processed_audio'])    # train_df
X_test = np.array(sample_test_df['processed_audio'])      # test_df
            # Sorties
# Convertir les étiquettes en tableaux numpy de tableaux numpy
y_train = np.vstack(sample_train_df['sequences'].to_numpy())  # train_df
y_test = np.vstack(sample_test_df['sequences'].to_numpy())    # test_df

# Reshape les datas d'entrée
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Fonction de création du modèle pour KerasClassifier
def create_model():
    layers = tf.keras.layers
    models = tf.keras.models

    model = models.Sequential()
    model.add(layers.Embedding(input_dim=vocab_size_source, output_dim=256, input_length=max_sequence_length_source))
    model.add(layers.LSTM(256))
    model.add(layers.RepeatVector(max_sequence_length_target))  # Répéter le vecteur de sortie pour chaque mot de la séquence cible
    model.add(layers.LSTM(256, return_sequences=True))  # Utiliser return_sequences=True pour obtenir une séquence en sortie
    model.add(layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax')))

    # Compiler le modèle
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

print('\nCreation du modèle...\n')
# Créer l'objet KerasClassifier avec la fonction de création de modèle
keras_model = KerasClassifier(build_fn=create_model,verbose=1)

print('\nCross validation & Entraînement du modèle avec recherche par grille...\n')
# Définir la grille de paramètres
params = {
    'batch_size': [16, 32, 64],
    'epochs': [10, 20, 30],
}

# Initialiser l'objet GridSearchCV avec le modèle KerasClassifier
grid_search = GridSearchCV(estimator=keras_model, param_grid=params,refit=False, cv=3, scoring='accuracy')      

# Entraîner le modèle avec recherche par grille
best_model = grid_search.fit(X_train,y_train)

# Meilleurs résultats
best_params = best_model.best_params_
best_score = best_model.best_score_
best_estimator = best_model.best_estimator_
print("\nMeilleurs paramètres trouvés :", best_params,'\n')
print("\nScore moyen sur les plis de validation :", best_score,'\n')
print("\nModèle entraîné avec la meilleure combinaison de paramètres :", best_estimator,'\n')

print('\n Evaluation du meilleur modèle sur les données de test ...\n')

# Évaluer le modèle sur l'ensemble de test
predictions = best_model.predict(X_test)
# Métrique d'évaluation
wer_score = wer(y_test, predictions)             
print("WER (Word Error Rate) :", wer_score)

print('\n Sauvegarde du modèle ...\n')
# Enregistrer le modèle au format .h5
best_model.save(f'{path_2}LTSM.h5')

print('Modèle sauvegardé.\n')