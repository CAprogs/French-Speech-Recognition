import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_models as tfm
import tf_models_official as tfm_off
from keras.layers import Conv1D, Dense, InputLayer
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow_addons.seq2seq import CTCLoss
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from keras.layers import MultiHeadAttention, Dropout, LayerNormalization
from jiwer import wer

# Chemin vers le fichier contenant les datas
treat_dataset_path = '//Users//charles-albert//Desktop/Projet Ingénieur//treat_datas'

# Charger les données prétraitées
train_data = pd.read_csv(f"{treat_dataset_path}train.tsv", sep="\t")
test_data = pd.read_csv(f"{treat_dataset_path}test.tsv", sep="\t")

# Paramètres du modèle
input_shape = (None, feature_dim)  # Dimension de l'entrée (à définir en fonction de tes données)
num_classes = len(vocab)  # Nombre de classes (lettres + tokens spéciaux)
num_filters = 64  # Nombre de filtres pour les couches CNN
num_heads = 4  # Nombre de têtes dans la couche Multi-Head Attention
dff = 256  # Dimension de la couche Feed Forward
dropout_rate = 0.1  # Taux de dropout

def create_model():
    # Entrée pour les caractéristiques audio
    audio_input = tf.keras.layers.InputLayer(input_shape=input_shape)
    
    # Couche Transformer Encoder
    encoder_output = tfm.nlp.layers.TransformerEncoderBlock(num_attention_heads=num_heads, inner_dim=dff, output_range=None,
                                             dropout_rate=dropout_rate)(audio_input)
    
    # Couche Transformer Decoder
    decoder_output = tfm.nlp.layers.TransformerDecoderBlock(num_attention_heads=num_heads, intermediate_size=dff,
                                             intermediate_activation='relu', dropout_rate=dropout_rate)(encoder_output)
    
    # Couche CNN
    cnn_output = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=3, padding='same', activation='relu')(decoder_output)

    # Couche Dense pour la classification des caractères
    dense_output = Dense(num_classes, activation='softmax')(cnn_output)
    
    # Création du modèle
    model = Model(inputs=audio_input, outputs=dense_output, name="TC_ASR")
    
    return model

# Création du modèle pour la recherche par grille
model = KerasClassifier(model=create_model, verbose=1)
hist = model.fit().history_
losses = hist["mean_absolute_error"]

# Paramètres à optimiser
param_grid = {
    'num_heads': [2, 4, 8],
    'dff': [256, 512, 1024],
    'dropout_rate': [0.1, 0.2, 0.3]
}

# Recherche par grille
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,refit=False, cv=3, verbose=3, scoring='accuracy')
grid_search.fit(train_data['audio'], train_data['transcription'])

# Meilleurs résultats
best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_estimator = grid_search.best_estimator_
print("\nMeilleurs paramètres trouvés :", best_params,'\n')
print("\nScore moyen sur les plis de validation :", best_score,'\n')
print("\nModèle entraîné avec la meilleure combinaison de paramètres :", best_estimator,'\n')

# Création du modèle avec les meilleurs paramètres
best_model = create_model()
best_model.summary()

# Compilation du modèle
optimizer = Adam(learning_rate=0.001)
loss = CTCLoss(blank_index=num_classes - 1)  # Utilisation de CTCLoss avec l'index du jeton vierge
best_model.compile(optimizer=optimizer, loss=loss)

# Entraînement du meilleur modèle
print("\nEntraînement du modèle...")
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("TC_ASR.h5", save_best_only=True)
]

best_model.fit(train_data['path'], train_data['sentence'],
               validation_data=(test_data['path'], test_data['sentence']),
               epochs=10, batch_size=32, callbacks=callbacks)

# Évaluation finale sur les données de test
print("\nEvaluation du modèle...")
predictions = best_model.predict(test_data['path'])
wer_score = wer(test_data['sentence'], predictions)
print("WER (Word Error Rate) :", wer_score)