import time
import numpy as np
import cv2
from scipy.ndimage import maximum_filter

from Server import Server, get_ip
from utils import Canny, HoughLines, CannyJIT

server = Server()

print(get_ip())

server.start()

cv2.namedWindow("original")
# cv2.namedWindow("canny")

while True:
    if not server.hasNewImg:
        cv2.waitKey(10)
        continue

    img = server.get_img()
    # canny_truth = cv2.Canny(img, 100, 200)
    canny = CannyJIT(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(np.uint8)

    cv2.imshow("canny", canny)
    # cv2.imshow("canny truth", canny_truth)

    hough = HoughLines(canny)
    max_filter = maximum_filter(hough[0], size=7)

    # cv2.imshow("hist", hough[0] / np.max(hough[0]))

    mat = np.repeat(hough[0] / max(np.max(hough[0]), 0.01), 3).reshape((*hough[0].shape, 3))
    mat[(max_filter == hough[0]) & (max_filter >= 200), 0] = 0
    mat[(max_filter == hough[0]) & (max_filter >= 200), 1] = 0
    mat[(max_filter == hough[0]) & (max_filter >= 200), 2] = 255

    mat = cv2.resize(mat, (1000, 1000))

    cv2.imshow("hist2", mat)

    lines = np.array(np.where((max_filter == hough[0]) & (max_filter >= 200))).transpose()
    # lines = cv2.HoughLines(canny, 1, np.pi/180, 200)

    if lines is not None:
        for line in lines:
            rho, theta = line
            rho = hough[1][rho]
            theta = hough[2][theta]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 3000 * (-b))
            y1 = int(y0 + 3000 * (a))
            x2 = int(x0 - 3000 * (-b))
            y2 = int(y0 - 3000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imshow("original", img)
    # cv2.imshow("canny", canny)

    cv2.waitKey(10)
