# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 13:14:22 2021

@author: steven
"""
import os 
import pandas as pd
import random
from skimage import io
import cv2
from matplotlib import pyplot as plt
import numpy as np
from shutil import copyfile
import sys

main_path = os.path.join("/home/uaodeepia/steven")

#Asignando 3 variables a las tablas de datos separados por coma de openimage
img_box_nombre = os.path.join(main_path, 'od_distanciamiento_social/train-images-boxable-with-rotation.csv')
anota_box_nombre = os.path.join(main_path,'od_distanciamiento_social/train-annotations-bbox.csv')
clase_desc_nombre = os.path.join(main_path,'od_distanciamiento_social/class-descriptions-boxable.csv')

#Comprendiendo como son los encabezados de cada tabla pasada a dataframe
img_box_pd = pd.read_csv(img_box_nombre)

anota_box_pd = pd.read_csv(anota_box_nombre)

#Para este caso toca  asegurarse de no tomar en cuenta encabezado por que no tiene
clas_desc_pd = pd.read_csv(clase_desc_nombre, header=None)

#Este código es irrelevante, solo sirve para tomar aleatoriamente imagenes a partir de su ID
test_img_id = anota_box_pd["ImageID"].value_counts().head(100).index.values
test_img_id = random.sample(list(test_img_id),1)


#Tomemos aleatoriamente 100 imagenes de personas y 100 de carros
#Lineas de clases
persona_pd =clas_desc_pd[clas_desc_pd[1]=='Person']
#Etiquetas de clases (Estas etiquetas serán primordiales cuando se vaya a copiar a disco local)
persona_label_pd = persona_pd[0].values[0]
#Usemos las etiquetas para bajar las box de anotaciones
persona_box_an= anota_box_pd[anota_box_pd['LabelName']==persona_label_pd]
#Bajemos los ID de las clases
persona_id=persona_box_an['ImageID']
#Dado que pueden haber varias personas en una imagen, podrían aparecer ID's repetidos. Vamos a usar una sola
persona_id = np.unique(persona_id)
#******Escoger aleatoriamente 3000 imagenes de cada clase
n=1000
sub_persona_id =random.sample(list(persona_id),n)

sub_persona_pd = img_box_pd.loc[img_box_pd['ImageID'].isin(sub_persona_id)]
#sub_persona_pd.shape

#Convirtiendo los dos subgrupos de imagenes en diccionarios.
sub_persona_dict =sub_persona_pd[['ImageID', 'OriginalURL']].set_index('ImageID')["OriginalURL"].to_dict()


clases = [sub_persona_dict]

person_path= os.path.join(main_path, "Person")

os.mkdir(person_path)

images_person_path= person_path

"""Salvando las imagenes a nuestros directorios locales. Observe el uso de "try" porque 
es posible que alguna dirección ya no exista"""
c=0
for i,o in enumerate([images_person_path]):
  errores=0
  
  for img_id, url in clases[i].items():
    try:
      c += 1
      print("guardando img {}".format(c))
      img = io.imread(url)
      ruta = o +'/'+img_id+".jpg"
      io.imsave(ruta,img)
    except Exception as e:
      errores+=1
  print(f"Imagenes perdidas:{errores}")


#Creemos los directorios de train y test en nuestro drive
train_images = os.path.join(main_path, "train3000")
test_images = os.path.join(main_path, "test3000")

os.mkdir(train_images)
os.mkdir(test_images)

train_path= train_images
test_path= test_images


"""Este codigo nos permite tomar el 80% de las 200 imagenes (carros y personas) y
copiarlas a la carpeta train. El 20% restante copiadas a la carpeta test."""
clas=['Person']


for i in range(len(clas)):
  clase_path = os.path.join(main_path, clas[i])
  imgs = os.listdir(clase_path)
  random.shuffle(imgs)
  percent = int(n*0.8)
  train_imgs=imgs[:percent]
  test_imgs = imgs[percent:]

  for f in range(len(train_imgs)):
    src = os.path.join(clase_path,train_imgs[f])
    dst = os.path.join(train_path,train_imgs[f])
    copyfile(src,dst)
  for f in range(len(test_imgs)):
    src = os.path.join(clase_path,test_imgs[f])
    dst = os.path.join(test_path,test_imgs[f])
    copyfile(src,dst)
#print(imgs)
    
"""Aqui creamos un dataframe de pandas con la información
que es necesaria para nuestro entrenamiento: nombre del archivo imagen,
x1,x2,y1,y2 (datos de la box) y la etiqueta de clase (Persona o Carro).""" 
etiquetas =[persona_label_pd]
train_df = pd.DataFrame(columns=['Nombre','Xmin','Xmax','Ymin', 'Ymax','Label'])

train_imgs = os.listdir(train_path)
c=0
for i in range(len(train_imgs)):
  c=c+1
  print("imagen procesada para train: {}".format(c))
  sys.stdout.flush()
  img_nombre = train_imgs[i]
  img_id=img_nombre[0:16]
  tmp_df = anota_box_pd[anota_box_pd['ImageID']==img_id]
  for index, row in tmp_df.iterrows():
    labelName = row['LabelName']
    for i in range(len(etiquetas)):
      if labelName == etiquetas[i]:
        train_df = train_df.append({'Nombre':img_nombre,
                                    'Xmin':row['XMin'],
                                    'Xmax':row['XMax'],
                                    'Ymin':row['YMin'],
                                    'Ymax':row['YMax'],
                                    'Label':clas[i]},
                                   ignore_index=True)
        
#Realizando lo mismo para la carpeta de test.
test_df = pd.DataFrame(columns=['Nombre','Xmin','Xmax','Ymin', 'Ymax','Label'])

test_imgs = os.listdir(test_path)
c=0
for i in range(len(test_imgs)):
  sys.stdout.flush()
  c=c+1
  print("imagen procesada para test: {}".format(c))
  img_nombre = test_imgs[i]
  img_id=img_nombre[0:16]
  tmp_df = anota_box_pd[anota_box_pd['ImageID']==img_id]
  for index, row in tmp_df.iterrows():
    labelName = row['LabelName']
    for i in range(len(etiquetas)):
      if labelName == etiquetas[i]:
        test_df = test_df.append({'Nombre':img_nombre,
                                    'Xmin':row['XMin'],
                                    'Xmax':row['XMax'],
                                    'Ymin':row['YMin'],
                                    'Ymax':row['YMax'],
                                    'Label':clas[i]},
                                   ignore_index=True)
        
#convirtiendo los dataframes creados a archivos CSV.
train_df.to_csv('train.csv')
test_df.to_csv('test.csv')


"""Este codigo nos sirve para crear un archivo texto donde estara la informacion
de cada imagen de entrenamiento con sus datos de bbox y etiqueta.
Observe como cada linea es un bbox diferente. El archivo de imagen se puede repetir
tantas veces como bbox tenga la imagen."""
train_df = pd.read_csv('train.csv')
c=0
with open(main_path + "/od_distanciamiento_social/entrenamiento3000.txt", 'w+') as f:
  for idx, row in train_df.iterrows():
    c += 1
    print("Agregando imagen {} al archivo".format(c))
    archivo = os.path.join(train_path,row['Nombre'])
    img = cv2.imread(archivo)
    h,w = img.shape[:2]
    x1 = int(row['Xmin']*w)
    x2 = int(row['Xmax']*w)
    y1 = int(row['Ymin']*h)
    y2 = int(row['Ymax']*h)
    
    label = row['Label']
    f.write(archivo + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + label + '\n')


#Igualmente para el archivo test.

test_df = pd.read_csv('test.csv')
c=0
with open(main_path + "/od_distanciamiento_social/test3000.txt", 'w+') as f:
  for idx, row in test_df.iterrows():
    c += 1
    print("Agregando imagen {} al archivo".format(c))
    archivo = os.path.join(test_path,row['Nombre'])
    img = cv2.imread(archivo)
    h,w = img.shape[:2]
    x1 = int(row['Xmin']*w)
    x2 = int(row['Xmax']*w)
    y1 = int(row['Ymin']*h)
    y2 = int(row['Ymax']*h)
    
    label = row['Label']
    f.write(archivo + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + label + '\n')
