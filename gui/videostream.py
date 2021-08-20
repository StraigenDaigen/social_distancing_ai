import threading
import time
import logging

import cv2


class VideoStream:

    def __init__(self, source):

        self.source = source
        self.stream = cv2.VideoCapture(self.source)
        self.grabbed, self.frame = None, None
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, name="video_stream", args=())
        t.daemon = True
        t.start()
        return self

    def read(self):

        if not self.grabbed:
            return None, None


        return self.grabbed, self.frame

    def update(self):

        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()

            #print(self.grabbed)
        self.stream.release()

    def stop(self):
        self.stopped = True
