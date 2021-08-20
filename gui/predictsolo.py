# -*- coding: utf-8 -*-
"""
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
from videostream import VideoStream
import threading

class Prediction:
    def __init__(self, path):
        super().__init__()
        label_path = "../retinanet_classes.csv"
        self.base_path = "../keras-retinanet"
        self.model_dir = "../retinanet_weights_person_uao_v1.h5"
        self.min_confidence = 0.7
        self.video_dir = path
        self.video = VideoStream(self.video_dir)
        self.video.start()
        time.sleep(0.4)
        self.frame = None
        self.grab = True
        self.frame_final = None
        self.model = None
        self.output= None
        self.image_rgb = None
        self.stopped = False
        self.LABELS = open(label_path).read().strip().split("\n")
        self.LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in self.LABELS}
        self.model = models.load_model(self.model_dir, backbone_name="resnet50")
        print('.......................VIENDO EL MODELO....................')

        area_real_pts = np.array([[265, 69], [402, 64], [673, 374], [43, 396]])

        # Valores para centrar el mapa en la ventana
        self.w_mapa_1 = int((800 / 2) + (200))  # (1266/4))
        self.w_mapa_2 = int((800 / 2) - (200))  # (1266/4))

        self.area_real_pts = np.array([[265, 69], [402, 64], [673, 374], [43, 396]])

        # PERPECTIVE TRANSFORM Y WARP PERSPECTIVE

        src_pts = np.array([[265, 69], [402, 64], [673, 374], [43, 396]], dtype=np.float32)
        dst_pts = np.array([[self.w_mapa_2, 0], [self.w_mapa_1, 0], [self.w_mapa_1, 710], [self.w_mapa_2, 710]],
                           dtype=np.float32)

        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    def ResizeWithAspectRatio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
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

    def trans_perspective(self):
        self.frame = imutils.resize(self.frame, width=800)

        

        imgAux = np.zeros(shape=(self.frame.shape[:2]), dtype=np.uint8)
        # imgAux = cv2.drawContours(imgAux,[area_mapa_pts],-1,(255),-1)
        imgAux = cv2.drawContours(imgAux, [self.area_real_pts], -1, (255), -1)
        warp = cv2.warpPerspective(imgAux, self.M, (self.w_mapa_1 + self.w_mapa_2, 710))

        image_test = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        return image_test,warp

    def predict(self,image):
        image = cv2.circle(image, (355, 109), 40, (255, 0, 0), -1)

        

        (boxes, scores, labels) = self.model.predict_on_batch(image)
        return boxes,scores,labels

    def start_visualizar(self):
        print('............thread start...........')

        t = threading.Thread(target=self.visualizar, name="predicted", args=())
        t.daemon = True
        t.start()
        return self

    def visualizar(self):
        while not self.stopped:
            if self.image_rgb is not None:
                print(np.shape(self.image_rgb))
                h, w, _ = np.shape(self.image_rgb)
                if h != 0 and w != 0:
                    print("visualizando")
                    #cv2.imwrite("prueba.jpg", self.image_rgb)
                    return self.image_rgb
    def stop(self):
        self.stopped = True


    def distanciamiento(self, boxes, scores, labels, warp):
        puntos_real = []
        puntos_p = []
        cantidad_personas = 0

        for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
            # filter out weak detections
            if score < self.min_confidence:
                continue

            # convert the bounding box coordinates from floats to integers
            box = box.astype("int")

            # build the label and draw the label + bounding box on the output
            # image
            label = "{}: {:.2f}".format(self.LABELS[label], score)
            cv2.rectangle(self.output, (box[0], box[1]), (box[2], box[3]),
                          (0, 255, 0), 2)
            # Obtener punto central inferior del bounding box
            pto_x = box[0] + ((box[2] - box[0]) / 2)
            pto_y = box[3]
            p_real = (int(pto_x), pto_y)
            # print(p_real)
            puntos_real.append(p_real)

            cv2.circle(self.output, (int(pto_x), pto_y), 5, (0, 0, 255), -1)
            cv2.putText(self.output, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            p_mapa_x = (self.M[0][0] * pto_x + self.M[0][1] * pto_y + self.M[0][2]) / (
                (self.M[2][0] * pto_x + self.M[2][1] * pto_y + self.M[2][2]))
            p_mapa_y = (self.M[1][0] * pto_x + self.M[1][1] * pto_y + self.M[1][2]) / (
                (self.M[2][0] * pto_x + self.M[2][1] * pto_y + self.M[2][2]))
            p_mapa = (int(p_mapa_x), int(p_mapa_y))
            cv2.circle(warp, p_mapa, 5, (0, 255, 0), -1)
            puntos_p.append(p_mapa)
            cantidad_personas += 1

        (alto, ancho) = self.output.shape[:2]
        if len(puntos_p) > 1:

            for punto1, punto2 in itertools.combinations(puntos_p, 2):
                s = time.time()
                x_p_trans = puntos_p.index(punto1)
                y_p_trans = puntos_p.index(punto2)
                # cv2.line(output, puntos_real[x_p_trans], puntos_real[y_p_trans], [155, 133, 0], 1)
                distancia = dist.euclidean(punto1, punto2)
                # print("PUNTO 1: "+str(point1)+" PUNTO 2: "+str(point2)+" DISTANCIA: "+str(distancia))

                if distancia < 75:
                    # print(x_p_trans,' ',y_p_trans)
                    cv2.line(self.output, puntos_real[x_p_trans], puntos_real[y_p_trans], [255, 0, ], 2)
                    (alto, ancho) = self.output.shape[:2]
                    peligro = "PELIGRO DE CONTAGIO"
                    cv2.putText(self.output, peligro, (int(alto * 0.55), int(ancho * 0.55)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
               # print("tiempo de comparar todas las distancias: {}".format(time.time() - s))
        puntos_real.clear()
        puntos_p.clear()

        return cantidad_personas, alto, ancho

    def  read(self):
        self.video = cv2.VideoCapture(self.video_dir)
        time.sleep(0.3)
        frames = 0


        while True:
            c=1
            self.grab, self.frame = self.video.read()
            if c>frames:
                if self.grab == True:
                    image,warp = self.trans_perspective()
                    self.output = image.copy()
                    image = preprocess_image(image)
                    (image, scale) = resize_image(image)
                    #Circle in the publicity to avoid detect a fake person.
                    image = cv2.circle(image,(355,109),8,(205,205,205),-1)
                    image = np.expand_dims(image, axis=0)
                    (boxes, scores, labels) = self.predict(image)
                    boxes /= scale
                    cantidad_personas, alto, ancho = self.distanciamiento(boxes, scores, labels, warp)

                    # loop over the detections

                    # Cantidad de personas
                    aforo = "Aforo: " + str(cantidad_personas)
                    cv2.putText(self.output, aforo, (int(alto * 0.1), int(ancho * 0.55)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    self.image_rgb = cv2.cvtColor(self.output, cv2.COLOR_BGR2RGB)

                    # show our detected people

                    #warpr = self.ResizeWithAspectRatio(warp, width=420)
                    cv2.drawContours(self.image_rgb, [self.area_real_pts], -1, (0, 0, 0), 2)

                    print(type(self.image_rgb))
                    return(self.image_rgb)
                    #c+=1
                #cv2.imshow("CONTROL DE DISTANCIAMIENTO SOCIAL", image_rgb)
                # cv2.imshow("MAPA DEL PLANO (Top view)",warpr)

    def returned(self):


        self.video.stop()
        self.stop()



#if __name__ == '__main__':
    # Iniciar la aplicacion

 #   prediction = Prediction('campanario.mp4')
 #   #prediction.predict()
 #   time.sleep(0.5)
 #   prediction.read()









