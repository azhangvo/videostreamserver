import time
import numpy as np
import cv2

from Server import Server, get_ip
from utils import Canny

server = Server()

print(get_ip())

server.start()

cv2.namedWindow("original")
cv2.namedWindow("canny")

while True:
    if not server.hasNewImg:
        time.sleep(0.05)
        continue

    img = server.get_img()
    # canny = cv2.Canny(img, 100, 200)
    canny = Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(np.uint8)

    cv2.imshow("original", img)
    cv2.imshow("canny", canny)

    cv2.waitKey(10)
