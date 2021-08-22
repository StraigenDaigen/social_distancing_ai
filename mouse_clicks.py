import cv2
import imutils
import numpy as np




circles = np.zeros((4,2), np.int32)
counter = 0

print(circles)

def mousePoints(event, x, y,flags, parameters):
    global circles, counter
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        circles[counter] = x, y
        counter = counter + 1
        print(circles)  

img = cv2.imread("time_square.jpg")
img =  imutils.resize(img, width=800)

while True:

    if counter>=4:

        width, height = img.shape[:2]


        area_real_pts = np.array([circles[0],circles[1],circles[2],circles[3]])
            
        #Valores para centrar el mapa en la ventana
        w_mapa_1 = int((width/2)+(height))#(1266/4))
        w_mapa_2 = int((width/2)-(height))#(1266/4))

        area_mapa_pts = np.array([[w_mapa_2,0],[w_mapa_1,0],[w_mapa_1,448],[w_mapa_2,448]])
            

            
        #PERPECTIVE TRANSFORM Y WARP PERSPECTIVE
            
        src_pts = np.array([circles[0],circles[1],circles[2],circles[3]], dtype=np.float32)
        dst_pts = np.array([[0,0],[width,0],[0,height],[width,height]], dtype=np.float32)
            
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        imgOutput = cv2.warpPerspective(img, M, (width, height))
        cv2.imshow("Output image", imgOutput)


    for x in range(0,4):
        cv2.circle(img, (circles[x][0], circles[x][1]), 3, (0, 255, 0), cv2.FILLED)
        
    cv2.imshow("Original image", img)
    cv2.setMouseCallback("Original image", mousePoints)


    cv2.waitKey(1)