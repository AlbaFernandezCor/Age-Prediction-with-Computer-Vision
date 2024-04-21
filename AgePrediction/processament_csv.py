# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:25:10 2024

@author: Usuario
"""
import csv
import os
from PIL import Image

def verificar_imagen(path):
    try:
        Image.open(os.path.join("C:/Users/Usuario/Documents/Age_Prediction_VC_1/Age_Prediction_VC/CACD2000", path))
        return True
    except (FileNotFoundError, OSError):
        return False
 
def procesar_csv(archivo_entrada, archivo_salida):
    with open(archivo_entrada, 'r') as csv_file:
        with open(archivo_salida, 'w', newline='') as salida_file:
            lector_csv = csv.reader(csv_file)
            escritor_csv = csv.writer(salida_file)
            # Leer encabezados del archivo de entrada
            encabezados = next(lector_csv)
            escritor_csv.writerow(encabezados)  # Escribir encabezados en el archivo de salida
            for indice, ruta, edad in lector_csv:
                if verificar_imagen(ruta):
                    escritor_csv.writerow([indice, ruta, edad])
 
archivo_entrada = "C:/Users/Usuario/Documents/Age_Prediction_VC_1/Age_Prediction_VC/AgePrediction/datasets/cacd_valid.csv"
archivo_salida = 'datos_validos.csv'
 
procesar_csv(archivo_entrada, archivo_salida)
print("Proceso completado. Se han escrito las líneas válidas en '{}'.".format(archivo_salida))