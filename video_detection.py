# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:15:56 2021

@author: steven, natali, felipe
"""
import imutils
import argparse
import cv2
from scipy.spatial import distance as dist
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import resize_image
from keras_retinanet import models
from matplotlib import pyplot as plt
import numpy as np
import time
import itertools



#Función para dismuniur la escala de la ventana del mapa       

#Función para dismuniur la escala de la ventana del mapa en top view       




def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help = "path to the (optional) video file")
ap.add_argument("-o", "--output", type=str, default="",
    help="path to (optional) output video file")
ap.add_argument("-f", "--frames", type=int, default=50,
    help="Frames per prediction")
args = vars(ap.parse_args())



labels= "retinanet_classes.csv"
base_path = "keras-retinanet"
model_dir="retinanet_weights_person_uao_v1.h5"

min_confidence=0.7

if not args.get("video", False):
    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(0)
    time.sleep(0.01)
# otherwise, load the video
else:
    vs = cv2.VideoCapture(args["video"])
    writer = None


# load the class label mappings
LABELS = open(labels).read().strip().split("\n")
LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in LABELS}

# load the model from disk
model = models.load_model(model_dir, backbone_name="resnet50")


area_real_pts = np.array([[265,69],[402,64],[673,374],[43,396]])
    
#Valores para centrar el mapa en la ventana
w_mapa_1 = int((800/2)+(200))#(1266/4))
w_mapa_2 = int((800/2)-(200))#(1266/4))

area_mapa_pts = np.array([[w_mapa_2,0],[w_mapa_1,0],[w_mapa_1,448],[w_mapa_2,448]])
    

    
#PERPECTIVE TRANSFORM Y WARP PERSPECTIVE
    
src_pts = np.array([[265,69],[402,64],[673,374],[43,396]], dtype=np.float32)
dst_pts = np.array([[w_mapa_2,0],[w_mapa_1,0],[w_mapa_1,710],[w_mapa_2,710]], dtype=np.float32)
    
M = cv2.getPerspectiveTransform(src_pts, dst_pts)



c=0
# keep looping
while True:
    
    s = time.time()
    # grab the current frame
    (grabbed, frame) = vs.read()
    print("tiempo de lectura: {}".format(time.time()-s))
    
    # if we are viewing a video and we did not grab a
    # frame, then we have reached the end of the video
    if vs and not grabbed:
        break

    # resize the frame and convert it to grayscale
    frame = imutils.resize(frame, width = 800)

    #Circle in the publicity to avoid detect a fake person.
    frame = cv2.circle(frame,(355,109),8,(205,205,205),-1)
    if c>args["frames"]:
    
        c=0
        s = time.time() 
        # resize the frame and convert it to grayscale
        #frame = imutils.resize(frame, width = 800)
        
        imgAux = np.zeros(shape=(frame.shape[:2]),dtype=np.uint8)
        #imgAux = cv2.drawContours(imgAux,[area_mapa_pts],-1,(255),-1) 
        imgAux = cv2.drawContours(imgAux,[area_real_pts],-1,(255),-1)
        warp = cv2.warpPerspective(imgAux, M, (w_mapa_1+w_mapa_2, 710))
        
        image_test= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # load the input image (in BGR order), clone it, and preprocess it
        image = image_test
        output = image.copy()
        image = preprocess_image(image)
        (image, scale) = resize_image(image)
        image = np.expand_dims(image, axis=0)

        print("tiempo de pre procesado: {}".format(time.time()-s))
        
        s = time.time()
        # detect objects in the input image and correct for the image scale
        (boxes, scores, labels) = model.predict_on_batch(image)
        boxes /= scale
        print("tiempo de predicción: {}".format(time.time()-s))

        
        puntos_real=[]
        puntos_p=[]
        cantidad_personas=0
        # loop over the detections
        for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
            # filter out weak detections
            if score < min_confidence:
                continue
        
            # convert the bounding box coordinates from floats to integers
            box = box.astype("int")
        
            # build the label and draw the label + bounding box on the output
            # image
            label = "{}: {:.2f}".format(LABELS[label], score)
            cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]),
                (0, 255, 0), 2)
            #Obtener punto central inferior del bounding box
            pto_x = box[0]+((box[2]-box[0])/2)
            pto_y = box[3]
            p_real=(int(pto_x),pto_y)
            #print(p_real)
            puntos_real.append(p_real)
            
            cv2.circle(output,(int(pto_x),pto_y),5,(0,0,255),-1)
            cv2.putText(output, label, (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            
            p_mapa_x = (M[0][0]*pto_x + M[0][1]*pto_y + M[0][2]) / ((M[2][0]*pto_x + M[2][1]*pto_y + M[2][2]))
            p_mapa_y = (M[1][0]*pto_x + M[1][1]*pto_y + M[1][2]) / ((M[2][0]*pto_x + M[2][1]*pto_y + M[2][2]))
            p_mapa = (int(p_mapa_x),int(p_mapa_y))
            cv2.circle(warp, p_mapa,5,(0,255,0),-1)
            puntos_p.append(p_mapa)
            cantidad_personas +=1
        
        if len(puntos_p)>1:
        

            for punto1, punto2 in itertools.combinations(puntos_p, 2): 
                s = time.time()
                x_p_trans = puntos_p.index(punto1)
                y_p_trans = puntos_p.index(punto2)
                cv2.line(output, puntos_real[x_p_trans], puntos_real[y_p_trans], [133, 133, 133], 1) 
                distancia=dist.euclidean(punto1,punto2)
                #print("PUNTO 1: "+str(point1)+" PUNTO 2: "+str(point2)+" DISTANCIA: "+str(distancia))
                
                if distancia < 75:
                    #print(x_p_trans,' ',y_p_trans)
                    cv2.line(output, puntos_real[x_p_trans], puntos_real[y_p_trans], [255, 0, ], 2) 
                    (alto, ancho)=output.shape[:2]
                    peligro="PELIGRO DE CONTAGIO"
                    cv2.putText(output, peligro, (int(alto*0.55), int(ancho*0.55)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                print("tiempo de comparar todas las distancias: {}".format(time.time()-s))

                    
        #Cantidad de personas
        aforo="Aforo: " + str(cantidad_personas)
        cv2.putText(output, aforo, (int(alto*0.1), int(ancho*0.55)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            
        
        image_rgb=cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
    
        # show our detected people     
        
        warpr = ResizeWithAspectRatio(warp, width=420)
        cv2.drawContours(image_rgb,[area_real_pts],-1,(0,0,0),2)    
        cv2.imshow("CONTROL DE DISTANCIAMIENTO SOCIAL", image_rgb)

        cv2.imshow("MAPA DEL PLANO (Top view)",warpr)

        
        puntos_real.clear()
        puntos_p.clear()
        
        #time.sleep(0)
        #cv2.imshow("VIDEO REAL", frame)

        key = cv2.waitKey(1) & 0xFF
    
        # if the `q` key was pressed, break from the loop
        if key == ord("q") or key==27:
            break
        
        if args["output"] != "" and writer is None:
            # initialize our video writer
            s = time.time()
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(args["output"], fourcc, 10,
                (image_rgb.shape[1], image_rgb.shape[0]), True)
            print("tiempo de escritura del video: {}".format(time.time()-s))

        # if the video writer is not None, write the frame to the output
        # video file

        if args["output"] != "":
            if writer is not None:
                writer.write(image_rgb)

    else:
        c += 1
        
        

    

    

        
# cleanup the camera and close any open windows
cv2.destroyAllWindows()
vs.release()