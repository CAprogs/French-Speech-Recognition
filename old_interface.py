import tkinter as tk
from tkvideo import tkvideo
from pickle import load
from tkinter import filedialog
import imageio.v3 as iio
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
import numpy as np
   
# Importation des données
print("Importation des données...\n")
# Importation de la normalisation
with open("ASR_NN//label_encoder.pkl","rb") as file:
    norm = load(file)
# Importation du mapping
with open("ASR_NN//scaler.pkl","rb") as file:
    mapping = load(file)
# Importation du modèle
model = tf.keras.saving.load_model("ASR_NN//LTSM.keras")

root = tk.Tk()

root.title("Automatic Speech Recognition - ECAM PROJECT")

# configurer le fond animé (GIF ou vidéo)
# afficher une vidéo :
video_path = "ASR_NN//background_vid.mp4"
video = iio.imread(video_path,plugin=None)

# Créer un widget Label pour afficher les images de la vidéo
label = tk.Label(root)
size = label.pack(expand=True, fill="both")
player =tkvideo(video_path,label,loop=1,size=size)
player.play()

######################################################################
# Créez une fonction pour afficher les images du lecteur vidéo en boucle
def animate(video):
    try:
        # Récupérer la prochaine image de la vidéo
        image = video.get_next_data()
        # Convertir l'image en format compatible avec Tkinter
        frame_image = tk.PhotoImage(data=image.tobytes())
        # Afficher l'image dans un widget Label
        label.config(image=frame_image)
        # Mettre à jour la référence à l'image pour éviter la suppression par le garbage collector
        label.image = frame_image
        # Appeler la fonction animate() pour passer à l'image suivante
        root.after(33, animate)  # Répéter l'opération toutes les 33 millisecondes (30 images par seconde)
    except:
        # En cas d'erreur ou de fin de la vidéo, arrêter l'animation
        print("Erreur")
        pass

# Créez une fonction pour gérer l'action du bouton d'enregistrement de la voix
def record_voice():
    duration = 5  # Durée de l'enregistrement en secondes
    fs = 44100  # Fréquence d'échantillonnage
    # Enregistrer la voix de l'utilisateur
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Attendre la fin de l'enregistrement
    # Sauvegarder l'enregistrement au format WAV
    file_path = "//Users//charles-albertkotto//Desktop//Deep Recognition Project//ASR_NN//User_audio//audio.wav"
    sf.write(file_path, recording, fs)

    pass

# Créez une fonction pour gérer l'action du bouton d'importation de fichier audio
def import_audio():
    # Ouvrir une boîte de dialogue de sélection de fichier
    file_path = filedialog.askopenfilename()
    # Traiter le fichier audio sélectionné
    if file_path:

        # -------- Traduction avec le modèle ----------
        print()

    else:
        pass

    pass
######################################################################

animate(video)

# Créez les boutons et associez les fonctions aux actions correspondantes
record_button = tk.Button(root, text="Enregistrer votre voix", command=record_voice)
import_button = tk.Button(root, text="Importer un fichier audio", command=import_audio)

# Positionnez les boutons sur l'interface utilisateur
record_button.pack(side=tk.LEFT, padx=10, pady=10)
import_button.pack(side=tk.LEFT, padx=10, pady=10)

# Créez une zone de texte pour afficher la traduction
translation_label = tk.Label(root, text="Traduction : ")
translation_label.pack()

# Créez une zone de texte pour afficher le pourcentage de confiance
confidence_label = tk.Label(root, text="Confiance : ")
confidence_label.pack(pady=10)

# Mettez à jour la zone de texte de la traduction avec le résultat généré
 # ---- Sortie du model ------
translation = "Text ..."
translation_label.config(text="Traduction : " + translation)

# Mettez à jour la zone de texte du pourcentage de confiance avec le pourcentage correspondant
confidence = 0.95 # ---- Moyenne de confidence du modèle ------
confidence_label.config(text="Confiance : " + str(confidence * 100) + "%")

root.mainloop()