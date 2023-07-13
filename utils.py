import math

import cv2
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter

FFT_CONVOLVE = True


# 1. reshape image into image cols
# 2. reshape kernel into linear shape
# 3. matrix multiply kernel and image cols
# 4. reshape result into convolved image
def convolve(mat, kernel):
    if FFT_CONVOLVE:
        return signal.fftconvolve(mat, kernel)
    kernel_size = kernel.shape[0]

    out_height = mat.shape[0] - kernel_size + 1
    out_width = mat.shape[1] - kernel_size + 1

    i0 = np.repeat(np.arange(kernel_size), kernel_size)
    i1 = np.repeat(np.arange(out_height), out_width)

    i = i0.reshape((-1, 1)) + i1.reshape((1, -1))

    j0 = np.tile(np.arange(kernel_size), kernel_size)
    j1 = np.tile(np.arange(out_width), out_height)

    j = j0.reshape((-1, 1)) + j1.reshape((1, -1))

    img_cols = mat[i, j]
    kernel_col = kernel.reshape((1, -1))

    return np.matmul(kernel_col, img_cols).reshape((out_height, out_width))


def naive_convolve(mat, kernel):
    kernel_size = kernel.shape[0]

    out_height = mat.shape[0] - kernel_size + 1
    out_width = mat.shape[1] - kernel_size + 1

    convolved_mat = np.zeros((out_height, out_width))

    for i in range(convolved_mat.shape[0]):
        for j in range(convolved_mat.shape[1]):
            convolved_mat[i][j] = np.sum(kernel * mat[i:i + kernel_size, j:j + kernel_size])

    return convolved_mat


def Gaussian(mat, kernel_size=5, sigma=1.0):
    assert kernel_size % 2 == 1 and kernel_size > 0
    k = (kernel_size - 1) / 2
    gaussian_filter = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    for i in range(kernel_size):
        for j in range(i + 1):
            val = 1 / (2 * np.pi * sigma * sigma) * math.exp(-(
                    (i + 1 - (k + 1)) * (i + 1 - (k + 1)) +
                    (j + 1 - (k + 1)) * (j + 1 - (k + 1))
            ) / 2 / sigma / sigma)
            gaussian_filter[i, j] = val
            gaussian_filter[j, i] = val

    return convolve(mat, gaussian_filter)


def Sobel(mat):
    x_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
    y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)

    convolved_x = convolve(mat, x_kernel)
    convolved_x[convolved_x == 0] = 1e-9
    convolved_y = convolve(mat, y_kernel)

    return np.sqrt(convolved_x * convolved_x + convolved_y * convolved_y), np.arctan(convolved_y / convolved_x)


def naive_nms(gradient, direction):
    def grad(y, x):
        if y < 0 or y >= gradient.shape[0] or x < 0 or x >= gradient.shape[1]:
            return 0
        return gradient[y, x]

    for i in range(gradient.shape[0]):
        for j in range(gradient.shape[1]):
            if 22.5 >= direction[i, j] > -22.5:
                if gradient[i, j] < grad(i, j + 1) or gradient[i, j] < grad(i, j - 1):
                    gradient[i, j] = 0
            elif 67.5 >= direction[i, j] > 22.5:
                if gradient[i, j] < grad(i - 1, j + 1) or gradient[i, j] < grad(i + 1, j - 1):
                    gradient[i, j] = 0
            elif -22.5 >= direction[i, j] > -67.5:
                if gradient[i, j] < grad(i + 1, j + 1) or gradient[i, j] < grad(i - 1, j - 1):
                    gradient[i, j] = 0
            else:
                if gradient[i, j] < grad(i + 1, j) or gradient[i, j] < grad(i - 1, j):
                    gradient[i, j] = 0

        return gradient


def Canny(mat):
    blurred = Gaussian(mat, kernel_size=5, sigma=1.4)
    gradient, direction = Sobel(blurred)

    # Gradient Magnitude Thresholding

    direction = np.round(direction / np.pi * 4) + 2

    g0a = np.roll(gradient, (0, 1), axis=(0, 1))
    g0b = np.roll(gradient, (0, -1), axis=(0, 1))
    g1a = np.roll(gradient, (-1, 1), axis=(0, 1))
    g1b = np.roll(gradient, (1, -1), axis=(0, 1))
    g2a = np.roll(gradient, (1, 0), axis=(0, 1))
    g2b = np.roll(gradient, (-1, 0), axis=(0, 1))
    g3a = np.roll(gradient, (1, 1), axis=(0, 1))
    g3b = np.roll(gradient, (-1, -1), axis=(0, 1))

    gradient[(direction == 2) & ((g0a > gradient) | (g0b > gradient))] = 0
    gradient[(direction == 3) & ((g1a > gradient) | (g1b > gradient))] = 0
    gradient[((direction == 0) | (direction == 4)) & ((g2a > gradient) | (g2b > gradient))] = 0
    gradient[(direction == 1) & ((g3a > gradient) | (g3b > gradient))] = 0

    # discretized_direction = np.zeros(direction.shape, dtype=np.uint8)

    # 0 ( -22.5 -  22.5 ) -> left to right
    # 1 (  22.5 -  67.5 ) -> northeast to southwest
    # 2 (  67.5 - 112.5 ) -> north to south
    # 3 ( 112.5 - 157.5 ) -> northwest to southeast
    # discretized_direction[(direction > np.pi / 8) & (direction < np.pi * 3 / 8)] = 1
    # discretized_direction[(direction > np.pi * 3 / 8) | (direction < -np.pi * 3 / 8)] = 2
    # discretized_direction[(direction < -np.pi / 8) & (direction > -np.pi * 3 / 8)] = 3

    # neighbor_directions = np.array([
    #     [[0, 1], [0, -1]],
    #     [[-1, 1], [1, -1]],
    #     [[1, 0], [-1, 0]],
    #     [[1, 1], [-1, -1]]
    # ])

    # Please don't look its 2am and my brain wont work

    # gradient_neighbors = np.array(list(zip(np.repeat(np.arange(gradient.shape[0]), gradient.shape[1]),
    #                                        np.tile(np.arange(gradient.shape[1]), gradient.shape[0])))).transpose()
    #
    # neighbors1 = gradient_neighbors + neighbor_directions[discretized_direction.flatten()][:, 0, :].transpose()
    # neighbors2 = gradient_neighbors + neighbor_directions[discretized_direction.flatten()][:, 1, :].transpose()
    #
    # neighbors1[0][neighbors1[0, :] < 0] = 0
    # neighbors1[0][neighbors1[0] >= gradient.shape[0]] = gradient.shape[0] - 1
    # neighbors1[1][neighbors1[1, :] < 0] = 0
    # neighbors1[1][neighbors1[1] >= gradient.shape[1]] = gradient.shape[1] - 1
    # neighbors2[0][neighbors2[0, :] < 0] = 0
    # neighbors2[0][neighbors2[0] >= gradient.shape[0]] = gradient.shape[0] - 1
    # neighbors2[1][neighbors2[1, :] < 0] = 0
    # neighbors2[1][neighbors2[1] >= gradient.shape[1]] = gradient.shape[1] - 1
    #
    # # gradient = gradient.flatten()
    # gradient[gradient < gradient[tuple(neighbors1)].reshape(gradient.shape)] = 0
    # gradient[gradient < gradient[tuple(neighbors2)].reshape(gradient.shape)] = 0

    # gradient[gradient >= 100] = 255
    # gradient[(gradient < 100) & (gradient > 30)] = 150

    # Double Thresholding

    edge_strength = np.zeros(gradient.shape, dtype=np.uint8)
    edge_strength[gradient >= 30] = 1
    edge_strength[gradient >= 100] = 2
    edge_strength = np.pad(edge_strength, 1)

    # Hysteresis

    q = np.array(np.where(edge_strength == 2)).transpose().tolist()
    directions = [[-1, 1], [0, 1], [1, 1], [-1, 0], [1, 0], [-1, -1], [0, -1], [1, -1]]
    while len(q):
        ele = q.pop()
        for dir in directions:
            if edge_strength[ele[0] + dir[0], ele[1] + dir[1]] == 1:
                edge_strength[ele[0] + dir[0], ele[1] + dir[1]] = 2
                q.append([ele[0] + dir[0], ele[1] + dir[1]])

    final = np.zeros(gradient.shape)
    final[edge_strength[1:-1, 1:-1] == 2] = 255

    return final


def HoughLines(edge_mat):
    edge_points = np.array(np.where(edge_mat == 255)).transpose()

    thetas = np.linspace(0, np.pi, 360)[:-1]
    rhos = np.matmul(edge_points, np.array([np.sin(thetas), np.cos(thetas)])).flatten()

    hist = np.histogram2d(rhos, np.tile(thetas, edge_points.shape[0]), bins=(5000, thetas.size))

    return hist, maximum_filter(hist[0], size=9)


if __name__ == '__main__':
    image = cv2.imread("./random_photo3.png")

    # cv2.imshow("original", image)

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(grayscale.shape)
    # cv2.imshow("grayscale", grayscale)

    # kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    # convolved_x = convolve(grayscale, kernel_x)
    # naive_convolved_x = naive_convolve(grayscale, kernel)
    #
    # np.testing.assert_array_equal(convolved_x, naive_convolved_x)

    # convolved_x[convolved_x < 0] = 0
    # convolved_x[convolved_x > 255] = 255
    # convolved_x = convolved_x.astype(np.uint8)

    # cv2.imshow("convolved x", convolved_x)

    # kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # convolved_y = convolve(grayscale, kernel_y)
    #
    # convolved_y[convolved_y < 0] = 0
    # convolved_y[convolved_y > 255] = 255
    # convolved_y = convolved_y.astype(np.uint8)

    # cv2.imshow("convolved y", convolved_y)

    # gaussian = Gaussian(grayscale, sigma=1.4)

    # cv2.imshow("gaussian", gaussian.astype(np.uint8))

    # sobel = Sobel(gaussian)
    #
    # sobel[0][sobel[0] > 255] = 255

    # cv2.imshow("sobel", sobel[0].astype(np.uint8))

    canny = Canny(grayscale)

    cv2.imshow("canny", canny.astype(np.uint8))

    canny_truth = cv2.Canny(grayscale, 400, 500)

    cv2.imshow("canny truth", canny_truth)

    cv2.imshow("grayscale", grayscale)

    hough = HoughLines(canny)

    cv2.imshow("hist", hough[0][0] / np.max(hough[0][0]))

    mat = np.repeat(hough[0][0] / np.max(hough[0][0]), 3).reshape((*hough[0][0].shape, 3))
    mat[(hough[1] == hough[0][0]) & (hough[1] >= 50), 0] = 0
    mat[(hough[1] == hough[0][0]) & (hough[1] >= 50), 1] = 0
    mat[(hough[1] == hough[0][0]) & (hough[1] >= 50), 2] = 255

    mat = cv2.resize(mat, (1000, 1000))

    cv2.imshow("hist2", mat)

    lines = np.array(np.where((hough[1] == hough[0][0]) & (hough[1] >= 50))).transpose()

    canny = cv2.cvtColor(canny.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for line in lines:
        rho, theta = line
        rho = hough[0][1][rho]
        theta = hough[0][2][theta]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 3000 * (-b))
        y1 = int(y0 + 3000 * (a))
        x2 = int(x0 - 3000 * (-b))
        y2 = int(y0 - 3000 * (a))
        cv2.line(canny, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imshow("test", canny)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
