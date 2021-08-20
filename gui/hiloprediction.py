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


class Prediction:
    def __init__(self, path):
        super().__init__()

        self.video_dir = path
        self.video = VideoStream(self.video_dir)
        self.video.start()
        time.sleep(0.3)

        label_path = "../retinanet_classes.csv"
        self.base_path = "../keras-retinanet"
        self.model_dir = "../retinanet_weights_person_uao_v1.h5"

        self.min_confidence = 0.7

        self.frame = None
        self.grab = True
        self.frame_final = None
        self.model = None
        self.output = None
        self.image_rgb = None
        self.LABELS = open(label_path).read().strip().split("\n")
        self.LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in self.LABELS}
        self.model = models.load_model(self.model_dir, backbone_name="resnet50")
        print('.......................MODELO....................')



        # Valores para centrar el mapa en la ventana
        self.w_mapa_1 = int((800 / 2) + (200))  # (1266/4))
        self.w_mapa_2 = int((800 / 2) - (200))  # (1266/4))

        self.area_real_pts = np.array([[265, 69], [402, 64], [673, 374], [43, 396]])

        # PERPECTIVE TRANSFORM Y WARP PERSPECTIVE

        src_pts = np.array([[265, 69], [402, 64], [673, 374], [43, 396]], dtype=np.float32)
        dst_pts = np.array([[self.w_mapa_2, 0], [self.w_mapa_1, 0], [self.w_mapa_1, 710], [self.w_mapa_2, 710]],
                           dtype=np.float32)

        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    def trans_perspective(self):
        self.frame = imutils.resize(self.frame, width=800)

        imgAux = np.zeros(shape=(self.frame.shape[:2]), dtype=np.uint8)
        imgAux = cv2.drawContours(imgAux, [self.area_real_pts], -1, (255), -1)
        warp = cv2.warpPerspective(imgAux, self.M, (self.w_mapa_1 + self.w_mapa_2, 710))
        image_test = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        return image_test, warp

    def predict(self, image):

        (boxes, scores, labels) = self.model.predict_on_batch(image)
        return boxes, scores, labels

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

    def read(self):

        frames = 1
        c = 0

        while True:
            self.grab, self.frame = self.video.read()
            if frames < c:
                c = 0
                if self.grab == True:
                    image, warp = self.trans_perspective()
                    self.output = image.copy()
                    image = preprocess_image(image)
                    (image, scale) = resize_image(image)
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

                    # warpr = self.ResizeWithAspectRatio(warp, width=420)
                    cv2.drawContours(self.image_rgb, [self.area_real_pts], -1, (0, 0, 0), 2)
                    # cv2.imshow("CONTROL DE DISTANCIAMIENTO SOCIAL", image_rgb)
                    # cv2.imshow("MAPA DEL PLANO (Top view)",warpr)
                    return (True, self.image_rgb)
                    print("returned", c)
                    time.sleep(0.5)
                    # return image_rgb
                else:
                    self.video.stop()

            c += 1

    def returned(self):
        if self.image_rgb is not None:
            print("images end")
            return (self.image_rgb)

# if __name__ == '__main__':
# Iniciar la aplicacion

# prediction = Prediction('campanario.mp4')

# time.sleep(0.5)
# prediction.read()
