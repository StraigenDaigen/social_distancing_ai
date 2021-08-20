import argparse
from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
import imutils

import time

#

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--hilos", type=str, default="",
    help="put 1 to activate threating")
args = vars(ap.parse_args())

if args["hilos"]=="":
    from predictionclass import Prediction
    
elif args["hilos"]=="1":
    from hiloprediction import Prediction
    
    

class Application(Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        self.video = None
        self.stopped = True


    def create_widgets(self):
        self.btnVisualizar = Button(self, text="Elegir y procesar video", command=self.visualizar_video)
        self.btnVisualizar.grid(column=0, row=0)

        self.btnNueva = Button(self, text="Nuevo video", command=self.nueva)
        self.btnNueva.grid(column=1, row=0, padx=5, pady=5)
        self.btnNueva["state"] = DISABLED

        self.labelInfo = Label(self, text="Video seleccionado:")
        self.labelInfo.grid(column=0, row=1)


        self.labelInfoVideoPath = Label(self, text="Aún no ha seleccionado ningún video")
        self.labelInfoVideoPath.grid(column=1, row=1)


        self.labelVideo = Label(self)
        self.labelVideo.grid(column=0, row=2, columnspan=2)


        self.labelVideo = Label(self)
        self.labelVideo.grid(column=0, row=2, columnspan=3)



    def visualizar(self):

        if self.video is not None:


            grab, frame = self.video.read()


            #if grab is True:


            frame = imutils.resize(frame, width=640)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)

            self.labelVideo.configure(image=img)
            self.labelVideo.image = img
            self.labelVideo.after(10, self.visualizar)


        else:
            self.stop()


    def inicialparameters(self):
        print("PARAMETERS INICIALES")
        self.labelVideo.image = ""
        self.labelInfoVideoPath.configure(text="Aún no ha seleccionado ningún video")
        self.video = None

    def stop(self):
        self.inicialparameters()
        self.stopped = True

    def nueva(self):
        self.btnVisualizar["state"] = ACTIVE
        self.btnNueva["state"] = DISABLED

        self.inicialparameters()



    def visualizar_video(self):
        if self.video is not None:
            self.inicialparameters()

        video_path = filedialog.askopenfilename(filetypes=[
            ("all video format", ".mp4"),
            ("all video format", ".avi")])


        if len(video_path) > 0:
            self.btnVisualizar["state"] = DISABLED
            self.btnNueva["state"] = ACTIVE

            self.labelInfoVideoPath.configure(text=video_path)
            self.video = Prediction(video_path)
            self.stopped = False
            self.visualizar()
        else:
            self.labelInfoVideoPath.configure(text="Aún no ha seleccionado ningún video")




root = tk.Tk()
app = Application(master=root)
app.mainloop()



