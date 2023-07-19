# Importer les bibliothèques nécessaires
print('Importation des librairies...\n')
import pandas as pd
import numpy as np
import re
import json
import os
import librosa
import random

############################################################################################################################
def Sup_columns(Dataframe):
    column_to_dtop = ['client_id', 'up_votes', 'down_votes', 'age', 'gender', 'accents', 'variant', 'locale', 'segment']
    Dataframe.drop(column_to_dtop, inplace=True, axis = 1)
    print(len(column_to_dtop),'colonnes supprimées.\n')
    return Dataframe

def add_fullaudio_path (Dataframe,clips_path):
    Dataframe['path'] = clips_path + '//' + Dataframe['path'] # Créer une nouvelle colonne 'path' en concaténant path_audio avec le contenu de 'path'
    new_df = Dataframe[['path', 'sentence']] # Créer un nouveau DataFrame avec les colonnes 'path' et 'sentence' uniquement
    # Afficher le premier chemin audio et la première phrase
    #print(new_df['path'][0])
    #print(new_df['sentence'][0])
    return new_df

"""
def treat_sentence(Dataframe):
    Dataframe["sentence"] = Dataframe["sentence"].apply(clean_sentence)
    # Afficher le premier chemin audio et la première phrase
    #print(Dataframe['path'][0])
    #print(Dataframe['sentence'][0])
    return Dataframe

def clean_sentence(text):
    text = re.sub(r"[:,-?!;.@#+*$£%<>_°)(&=\[\]\^\"]", "", text)  # Supprimer les caractères spéciaux
    text = text.lower()  # Normaliser en lettres minuscules
    return text

def create_vocab(Dataframe): # UNIQUEMENT SUR LE TRAIN DATASET
    
    print("Extraction du vocabulaire..")
    sentence_concatenated = "".join(Dataframe["sentence"])
    letters = sorted(list(set(sentence_concatenated)))  # Lettres distinctes
    letters.append("|")  # Ajouter le caractère spécial pour l'espace
    letters.append("<unk>")  # Ajouter le token "inconnu"
    letters.append("<blank>")  # Ajouter le token de remplissage

    vocab = {char: idx for idx, char in enumerate(letters)}
    print('\nVocabulaire extrait :',vocab)

    print("\nTokenization..")
    Dataframe["tokens"] = Dataframe["sentence"].apply(tokenize_text, args=(vocab,))
    

    with open(f"{new_dataset_path}vocabulaire.json", "w") as f:
        json.dump(vocab, f)
    
    return Dataframe

def tokenize_text(text, vocab):
    tokens = []
    for char in text:
        if char in vocab:
            tokens.append(vocab[char])
        else:
            tokens.append(vocab["<unk>"])  # Caractère inconnu
    return tokens
"""

def View_samplingRate(Dataframe):
    print("\nRandom sampling rate :")
    for i in range(0,4):
        r = random.randint(0,10000)
        signal, sr = librosa.load(Dataframe['path'][r], sr=None)  # Charger l'audio avec le sampling rate d'origine
        print(f"audio_file n°{r} : {sr}")

"""   
def process_audio(file_path, target_sr):
    signal, sr = librosa.load(file_path, sr=None)  # Charger l'audio avec le sampling rate d'origine
    if sr != target_sr:
        signal = librosa.resample(signal, sr, target_sr)  # Resampler l'audio au sampling rate cible

    return np.asarray(signal)

def resample_audio(Dataframe):
    target_sr = 16000  # Sampling rate cible

    # Parcourir tous les fichiers audio et les traiter
    for index, row in Dataframe.iterrows():
        audio_file = row["path"]
        processed_audio = process_audio(audio_file, target_sr)
""" 

############################################################################################################################

# Chemin vers les fichiers contenant les datas
dataset_path = '//Users//charles-albert//Desktop//Projet Ingénieur//datas//fr//'
clips_path = '//Users//charles-albert//Desktop//Projet Ingénieur//datas//fr//clips'
new_dataset_path = '//Users//charles-albert//Desktop/Projet Ingénieur//treat_datas//'

train_path = f'{dataset_path}train.tsv'
test_path = f'{dataset_path}test.tsv'

# Charger les fichiers TSV dans un DataFrame pandas
print('Creation des dataframes...\n')
train_df = pd.read_csv(train_path, delimiter='\t')
test_df = pd.read_csv(test_path, delimiter='\t')

print('\n',len(train_df),'Train samples trouvés\n')
train_df.info()

print('\n',len(test_df),'Test samples trouvés\n')
test_df.info()

# Supression des colonnes inutiles - Modification colonne 1 (path) - Modification colonne 2 (sentence)
print('\nTraitement des Datas...')
print('\nSupression des colonnes inutiles...')
train_df = Sup_columns(train_df)
test_df = Sup_columns(test_df)

print('\nTraitement de la colonne \'path\'...')
train_df = add_fullaudio_path(train_df,clips_path)
test_df = add_fullaudio_path(test_df,clips_path)

"""
print('\nTraitement de la colonne \'sentence\'...')
train_df = treat_sentence(train_df)
test_df = treat_sentence(test_df)

print('\nCreation d\'un vocabulaire et d\'un Tokenizer...')
train_df = create_vocab(train_df)
"""

print('\nVisualisation des Sampling Rate...')
View_samplingRate(train_df)
View_samplingRate(test_df)

print('\nVisualisation des nouveaux dataframes...')
print(train_df.head(5))
print(test_df.head(5))

print('\nSauvegarde...')
train_df.to_csv(f"{new_dataset_path}train.tsv", sep="\t", index=False)
test_df.to_csv(f"{new_dataset_path}test.tsv", sep="\t", index=False)
print('\nSauvegarde Terminée.')