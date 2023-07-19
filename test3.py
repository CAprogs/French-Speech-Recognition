print('Importation des librairies...\n')
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from jiwer import wer

path = '//Users//charles-albert//Desktop//Projet Ingénieur//treat_datas//'
path_2 = '//Users//charles-albert//Desktop//Projet Ingénieur//Test//'

# Chemin vers le dataset .TSV (validated audio)
dataset_path = f'{path}train.tsv'

# Lecture de la BDD
print('Chargement des dataframe ...\n')
df = pd.read_csv(dataset_path, delimiter='\t')
df.info()
#################################################### TEST sur un petit ensemble
# Sélectionner n éléments aléatoires sans modifier le DataFrame d'origine
sample_train_df = df.sample(n=280, replace=False)
sample_train_df.info()
####################################################

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

print('\nTraitement des caractères spéciaux ($,€ et %)...\n')
sample_train_df = replace(sample_train_df) 
sample_train_df.info()

print('Creation du Tokenizer...\n')
# Créer un tokenizer
tokenizer = Tokenizer(filters='!"#&()*+,-./:;<=>?@[\\]^_`{|}~',lower=True,split= " ",oov_token="<unk>") 
tokenizer.fit_on_texts(sample_train_df['sentence'])

# Convertir le texte en séquences numériques
train_sequences = tokenizer.texts_to_sequences(sample_train_df['sentence'])                                  

# Obtenir la taille du vocabulaire
vocab_size = len(tokenizer.word_index) + 1
print('Taille du vocabulaire :', vocab_size)

#obtenir la longueur maximale d'une phrase suivant la normalisation
max_sequence_length = 0
for i in train_sequences:
    if len(i) >= max_sequence_length:
        max_sequence_length = len(i)
    else:
        continue
print('\nmax senquence length : ',max_sequence_length) # longueur maximale uniforme

# Paddage des séquences pour qu'elles aient la même longueur
train_sequences_padded = pad_sequences(train_sequences, maxlen=max_sequence_length)

# Ajouter les séquences paddées dans une nouvelle colonne 'sequences' 
sample_train_df['sequences'] = train_sequences_padded.tolist()                                             
sample_train_df.info()

print('\nEnregistrement du Tokenizer...\n')
with open(f'{path_2}tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def data_transform(dataframe,features,max_length = 0): #labels
    print('\nExtraction de caractéristiques (mfcc & labels) ...\n')
    for index, row in dataframe.iterrows():
        fichier_audio = row['path']
        #phrase = row['sequences']

        # Charger le fichier audio avec librosa et extraire les caractéristiques
        audio, sr = librosa.load(fichier_audio)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr)

        # Normaliser les données
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

        # Mettre à jour la longueur maximale
        max_length = max(max_length, mfcc.shape[1])

        # traitement des phrases

        # Ajouter les caractéristiques et les labels aux listes ( les labels = phrases sous forme de tableau numpy )
        features.append(mfcc)
        #labels.append(phrase)

    # Remplir les matrices MFCC avec des zéros pour avoir la même dimension le long de l'axe 1
    for i in range(len(features)):
        features[i] = np.pad(features[i], ((0, 0), (0, max_length - features[i].shape[1])), mode='constant')

    """
    # Convertir les listes en tableaux numpy
    features = np.vstack(features)
    sample_train_df['sequences'] = features.tolist() liste de tableaux numpy dans le dataframe
    """
    # Redimensionner la matrice des étiquettes pour correspondre au nombre d'échantillons dans features
    #labels = np.array(labels)
    #labels = np.resize(labels, (features.shape[0],))

    return features#,labels

# Charger les fichiers audio et extraire les caractéristiques
features = [] # contient les caractéristiques extraites des fichiers audio, dans ce cas précis, les coefficients cepstraux en fréquence de Mel (MFCC).
#labels = [] # Chaque élément de cette liste correspond à la transcription associée à un fichier audio.

#features_,labels_ = data_transform(sample_train_df,features,labels)
features_ = data_transform(sample_train_df,features)
"""
features_ = pd.Series(features_)
print(features_)
"""
# Convertir chaque tableau NumPy de tableaux NumPy en une liste de tableaux NumPy
features_processed = [arr.tolist() for arr in features_]
# Création d'une série pandas à partir de la liste de tableaux NumPy
features_series = pd.Series(features_processed)
# Affichage de la série
print(features_series)
# Ajouter la série 'features' au DataFrame 'sample_train_df'
sample_train_df['features'] = features_series


# Diviser le dataset en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(features_, labels_, test_size=0.2, random_state=42)

"""
print('\nNormalisation ...\n')
# Encoder les labels en nombres entiers
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

print('\n Standardisation ...\n')
# Standardiser les données d'entraînement et de test
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
"""
"""
# Calcul score
print("Chargement des données d'entraînement et de test...\n")
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)

X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)
"""

# Fonction de création du modèle pour KerasClassifier
def create_model():
    print('\n Création du modèle ...\n')
    layers = tf.keras.layers
    models = tf.keras.models

    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(vocab_size, activation='softmax'))

    # Compiler le modèle
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Créer l'objet KerasClassifier avec la fonction de création de modèle
keras_model = KerasClassifier(build_fn=create_model)

print('\nCross validation & Entraînement du modèle avec recherche par grille...\n')
# Définir la grille de paramètres
param_grid = {
    'batch_size': [16, 32, 64],
    'epochs': [10, 20, 30],
}

# Initialiser l'objet GridSearchCV avec le modèle KerasClassifier
grid_search = GridSearchCV(keras_model, param_grid, scoring='accuracy', cv=3)

# Entraîner le modèle avec recherche par grille
grid_search.fit(X_train, y_train)

# Meilleurs résultats
best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_estimator = grid_search.best_estimator_
print("\nMeilleurs paramètres trouvés :", best_params,'\n')
print("\nScore moyen sur les plis de validation :", best_score,'\n')
print("\nModèle entraîné avec la meilleure combinaison de paramètres :", best_estimator,'\n')

"""
print('\n Evaluation du meilleur modèle sur les données de test ...\n')
# Évaluer le modèle sur l'ensemble de test
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print('Test Loss: ', test_loss)
print('Test Accuracy: ', test_accuracy)
"""

# Évaluer le modèle sur l'ensemble de test
predictions = grid_search.predict(X_test)
# Métrique d'évaluation
wer_score = wer(y_test, predictions)             
print("WER (Word Error Rate) :", wer_score)

print('\n Sauvegarde du modèle ...\n')
# Enregistrer le modèle au format .h5
grid_search.save(f'{path_2}MODEL.h5')


print('Données sauvegardées.\n')