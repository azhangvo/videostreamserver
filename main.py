import zmq
import socket
import cv2 as cv
import numpy as np

context = zmq.Context()
zmqsocket = context.socket(zmq.PAIR)
zmqsocket.bind("tcp://0.0.0.0:8001")

hostname = socket.getfqdn()
print("IP Address:", socket.gethostbyname_ex(hostname)[2][0])

message = zmqsocket.recv()
print("Received request: %s" % message)

zmqsocket.send(b"Success")

cv.namedWindow("Phone Camera")

try:
    while True:
        zmqsocket.send(b"D")
        data = zmqsocket.recv()

        img = cv.imdecode(np.frombuffer(data, dtype=np.uint8), cv.IMREAD_UNCHANGED)

        cv.imshow("Phone Camera", img)
        cv.waitKey(1)
except Exception as e:
    print(e)
finally:
    cv.destroyAllWindows()
