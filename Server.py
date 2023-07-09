import threading
import time

import zmq
import socket
import cv2 as cv
import numpy as np


def get_ip():
    hostname = socket.getfqdn()
    return socket.gethostbyname_ex(hostname)[2][0]


class Server:
    def __init__(self):
        self.context = zmq.Context()
        self.zmqSocket = None
        self.thread = None
        self.isRunning = False
        self.establishedConnection = False
        self.img = None
        self.hasNewImg = False

    def start(self):
        if self.isRunning:
            raise Exception("Server is already running")

        self.zmqSocket = self.context.socket(zmq.PAIR)
        self.zmqSocket.bind("tcp://0.0.0.0:8001")

        self.isRunning = True

        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

        self.zmqSocket.recv()
        self.zmqSocket.send(b"Success")

        self.establishedConnection = True

    def stop(self):
        if not self.isRunning:
            raise Exception("Server is not running")

        self.isRunning = False

    def loop(self):
        while not self.establishedConnection:
            time.sleep(0.1)
        while self.isRunning:
            self.zmqSocket.send(b"D")
            data = self.zmqSocket.recv()

            self.img = cv.imdecode(np.frombuffer(data, dtype=np.uint8), cv.IMREAD_UNCHANGED)

            self.hasNewImg = True
        self.zmqSocket.close()

    def get_img(self):
        self.hasNewImg = False
        return self.img

