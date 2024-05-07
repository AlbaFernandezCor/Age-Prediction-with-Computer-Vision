import shutil
import os

def copiar_imagenes(path_imagenes):

    destino = "C:/Users/Marina/Documents/GitHub/Uni/Age_Prediction_VC/ImgReduit/"

    if not os.path.exists(destino):
        os.makedirs(destino)
    
    # Iterar sobre cada imagen en el directorio de origen
    if os.path.isfile(path_imagenes):
        # Copiar la imagen al directorio de destino
        shutil.copy(path_imagenes, destino)
        print(f"Imagen {path_imagenes} copiada exitosamente a {destino}")

