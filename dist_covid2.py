import jetson.inference
import jetson.utils
import cv2
import argparse
import sys
import numpy as np
#import imutils
import itertools

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

# parse the command line
parser = argparse.ArgumentParser(description="Monitoreo de distanciamiento social por COVID-19")

parser.add_argument("input_URI", type=str, default="", nargs='?', help="Directorio del video de entrada")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="Directorio para guargar video")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="Escoger modelo pre-entrenado (las opciones están abajo)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="valor minimo de precisión para las detecciones")  

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)
cap = cv2.VideoCapture(opt.input_URI)

area_real_pts = np.array([[420,110],[642,100],[1069,594],[73,636]])
#Valores para centrar el mapa en la ventana
w_mapa_1 = int((800/2)+(200))#(1266/4))
w_mapa_2 = int((800/2)-(200))#(1266/4))

area_mapa_pts = np.array([[w_mapa_2,0],[w_mapa_1,0],[w_mapa_1,448],[w_mapa_2,448]])
#PERPECTIVE TRANSFORM Y WARP PERSPECTIVE
src_pts = np.array([[425,110],[642,100],[1069,594],[73,636]], dtype=np.float32)
dst_pts = np.array([[w_mapa_2,0],[w_mapa_1,0],[w_mapa_1,710],[w_mapa_2,710]], dtype=np.float32)
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
# process frames until the user exits
while True:
	ret,frame=cap.read()
	# capture the next image
	img = input.Capture()
	#frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	#frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
	#width = frame.shape[1]
	#height = frame.shape[0]
	#cuda_mem = jetson.utils.cudaFromNumpy(frame)
	# detect objects in the image (with overlay)
	detections = net.Detect(img, overlay=opt.overlay)
	#detections = net.Detect(cuda_mem, width, height)
	#_,_,center,_,_,_,_,_,_ = net.Detect(img, overlay=opt.overlay)
	#print("<<<<<DESDE AQUI>>>>")
	#print(detections[0].Center)
	#print("<<<<<HASTA AQUI>>>>")
	# print the detections
	print("detected {:d} objects in image".format(len(detections)))
	frame = np.array(img)
	#frame = imutils.resize(frame, width = 800)
	#frame = imutils.resize(frame, width = 800)
	imgb = cv2.cvtColor(frame,cv2.COLOR_RGBA2BGR)
	imgAux = np.zeros(shape=(imgb.shape[:2]),dtype=np.uint8)
	#imgAux = cv2.drawContours(imgAux,[area_mapa_pts],-1,(255),-1)
	imgAux = cv2.drawContours(imgAux,[area_real_pts],-1,(255),-1)
	warp = cv2.warpPerspective(imgAux, M, (w_mapa_1+w_mapa_2, 710))
	puntos_real=[]
	puntos_p=[]
	for detection in detections:
		pto_x = (detection.Center)[0]
		pto_y = ((detection.Center)[1])+((detection.Height)/2)
		p_real=(int(pto_x),int(pto_y))
		puntos_real.append(p_real)
		cv2.circle(imgb,(int(pto_x),int(pto_y)),5,(0,255,0),-1)

		p_mapa_x = (M[0][0]*pto_x + M[0][1]*pto_y + M[0][2]) / ((M[2][0]*pto_x + M[2][1]*pto_y + M[2][2]))
		p_mapa_y = (M[1][0]*pto_x + M[1][1]*pto_y + M[1][2]) / ((M[2][0]*pto_x + M[2][1]*pto_y + M[2][2]))
		p_mapa = (int(p_mapa_x),int(p_mapa_y))
		cv2.circle(warp, p_mapa,5,(0,255,0),-1)
		puntos_p.append(p_mapa)

	if len(puntos_p)>1:
		for punto1, punto2 in itertools.combinations(puntos_p, 2):
			x_p_trans = puntos_p.index(punto1)
			y_p_trans = puntos_p.index(punto2)
			cv2.line(imgb, puntos_real[x_p_trans], puntos_real[y_p_trans], [133, 133, 133], 1)
			#distancia=dist.euclidean(punto1,punto2)
			distancia = np.linalg.norm(np.array(punto1)-np.array(punto2))

			if distancia < 75:
				cv2.line(imgb, puntos_real[x_p_trans], puntos_real[y_p_trans], [0, 0, 255], 2)
				(alto, ancho)=imgb.shape[:2]
				peligro="PELIGRO DE CONTAGIO"
				cv2.putText(imgb, peligro, (int(alto*0.55), int(ancho*0.55)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	aforo="Aforo: " + str(len(detections))
	(alto, ancho)=imgb.shape[:2]
	cv2.putText(imgb, aforo, (int(alto*0.1), int(ancho*0.55)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)	

	# render the image
	#output.Render(img)
	#output.SetStatus("{:s} | Rendimiento {:.0f} FPS".format("DISTANCIAMIENTO DE PERSONAS", net.GetNetworkFPS()))
	cv2.drawContours(imgb,[area_real_pts],-1,(0,0,0),2)
	#image_rgb=cv2.cvtColor(imgb,cv2.COLOR_BGR2RGB)
	cv2.putText(imgb,"| Rendimiento {:.0f} FPS".format(net.GetNetworkFPS()) , (int(alto*1.2), int(ancho*0.55)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
	cv2.imshow("DISTANCIAMIENTO SOCIAL",imgb)
	warpr = ResizeWithAspectRatio(warp, width=420)
	cv2.imshow("MAPA DEL PLANO (Top view)",warpr)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q") or key==27:
		break
	# print out performance info
	net.PrintProfilerTimes()

	# exit on input/output EOS
	if not input.IsStreaming() or not output.IsStreaming():
		break
cap.release()
cv2.destroyAllWindows()