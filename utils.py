import math

from fast_histogram import histogram2d
from numba import jit, njit, float64, int64
import cv2
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter

FFT_CONVOLVE = True


# 1. reshape image into image cols
# 2. reshape kernel into linear shape
# 3. matrix multiply kernel and image cols
# 4. reshape result into convolved image
@jit(nopython=True)
def convolve(image, kernel):
    m, n = kernel.shape
    if m == n:
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y, x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i + m, j:j + m] * kernel)
    return new_image


# def convolve(mat, kernel):
#     return np.convolv(mat, kernel)
# if FFT_CONVOLVE:
#     return signal.fftconvolve(mat, kernel)
# kernel_size = kernel.shape[0]
#
# out_height = mat.shape[0] - kernel_size + 1
# out_width = mat.shape[1] - kernel_size + 1
#
# i0 = np.repeat(np.arange(kernel_size), kernel_size)
# i1 = np.repeat(np.arange(out_height), out_width)
#
# i = i0.reshape((-1, 1)) + i1.reshape((1, -1))
#
# # j0 = np.tile(np.arange(kernel_size), kernel_size)
# # j1 = np.tile(np.arange(out_width), out_height)
# # j0 = np.array([*np.arange(kernel_size)] * kernel_size)
# # j1 = np.array([*np.arange(out_width)] * out_height)
# j0 = np.repeat(np.arange(kernel_size), kernel_size).reshape((-1, kernel_size)).transpose().flatten()
# j1 = np.repeat(np.arange(out_width), out_height).reshape((-1, out_height)).transpose().flatten()
#
# j = j0.reshape((-1, 1)) + j1.reshape((1, -1))
#
# img_cols = mat[i, j]
# kernel_col = kernel.reshape((1, -1))
#
# return np.matmul(kernel_col, img_cols).reshape((out_height, out_width))


def naive_convolve(mat, kernel):
    kernel_size = kernel.shape[0]

    out_height = mat.shape[0] - kernel_size + 1
    out_width = mat.shape[1] - kernel_size + 1

    convolved_mat = np.zeros((out_height, out_width))

    for i in range(convolved_mat.shape[0]):
        for j in range(convolved_mat.shape[1]):
            convolved_mat[i][j] = np.sum(kernel * mat[i:i + kernel_size, j:j + kernel_size])

    return convolved_mat


@jit(nopython=True)
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


x_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)


@jit(float64[:, :](float64[:, :], int64, float64[:, :]), nopython=True)
def rnd1(x, decimals, out):
    return np.round_(x, decimals, out)


@jit(nopython=True)
def Sobel(mat):
    convolved_x = convolve(mat, x_kernel)
    # convolved_x[convolved_x == 0] = 1e-9
    for i in range(convolved_x.shape[0]):
        for j in range(convolved_x.shape[1]):
            if convolved_x[i, j] == 0:
                convolved_x[i, j] = 1e-9
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


@jit(nopython=True, cache=True)
def CannyJIT(mat):
    blurred = Gaussian(mat, kernel_size=5, sigma=1.4)
    gradient, direction = Sobel(blurred)

    direction *= 180 / np.pi

    for y in range(1, gradient.shape[0] - 1):  # Technically not reaching edges, need to fix later
        for x in range(1, gradient.shape[1] - 1):
            val = gradient[y, x]
            dir = direction[y, x]
            if (dir < -67.5 or dir > 67.5) and (val <= gradient[y + 1, x] or val <= gradient[y - 1, x]):
                gradient[y, x] = 0
            elif dir < -22.5 and (val <= gradient[y + 1, x + 1] or val <= gradient[y - 1, x - 1]):
                gradient[y, x] = 0
            elif dir < 22.5 and (val <= gradient[y, x + 1] or val <= gradient[y, x - 1]):
                gradient[y, x] = 0
            elif val <= gradient[y + 1, x - 1] or val <= gradient[y - 1, x + 1]:
                gradient[y, x] = 0

    edge_strength = np.zeros(gradient.shape, dtype=np.uint8)
    q = []
    for i in range(edge_strength.shape[0]):
        for j in range(edge_strength.shape[1]):
            if gradient[i, j] > 50:
                edge_strength[i, j] = 255
                q.append([i, j])
            elif gradient[i, j] > 10:
                edge_strength[i, j] = 1

    # Hysteresis

    directions = [[-1, 1], [0, 1], [1, 1], [-1, 0], [1, 0], [-1, -1], [0, -1], [1, -1]]
    while len(q):
        ele = q.pop()
        for dir in directions:
            ny = ele[0] + dir[0]
            nx = ele[1] + dir[1]
            if 0 <= ny < edge_strength.shape[0] and 0 <= nx < edge_strength.shape[1] and \
                    edge_strength[ny, nx] == 1:
                edge_strength[ny, nx] = 255
                q.append([ny, nx])

    return edge_strength


# @jit(nopython=True)
def Canny(mat):
    blurred = Gaussian(mat, kernel_size=5, sigma=1.4)
    gradient, direction = Sobel(blurred)

    # Gradient Magnitude Thresholding

    direction = np.zeros(direction.shape)
    rnd1(direction / np.pi * 4, 0, direction)
    direction += 2

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


# @jit(nopython=True)
# def HoughLines(edge_mat):
#     thetas = np.linspace(0, np.pi, 360)[:-1]
#     edge_points = []
#
#     for y in range(edge_mat.shape[0]):
#         for x in range(edge_mat.shape[1]):
#             if edge_mat[y, x] == 255:
#                 edge_points.append([y, x])
#
#     edge_points = np.array(edge_points)
#     # edge_points = np.array(np.where(edge_mat == 255)).transpose()
#
#     rhos = np.matmul(edge_points, np.array([np.sin(thetas), np.cos(thetas)])).flatten()
#
#     hist = np.histogram2d(rhos, np.tile(thetas, edge_points.shape[0]), bins=(5000, thetas.size))
#
#     return hist


def HoughLines(edge_mat):
    edge_points = np.array(np.where(edge_mat == 255)).transpose()

    thetas = np.linspace(0, np.pi, 360)[:-1]
    rhos = np.matmul(edge_points, np.array([np.sin(thetas), np.cos(thetas)])).flatten()

    hist = (
        histogram2d(rhos, np.tile(thetas, edge_points.shape[0]),
                    range=((np.min(rhos), np.max(rhos)), (0, np.max(thetas))),
                    bins=(5000, thetas.size)),
        np.linspace(np.min(rhos), np.max(rhos), 5000),
        thetas)

    return hist


# @jit(nopython=True)
# def houghLines(edges, dTheta, threshold):
#     imageShape = edges.shape
#     imageDiameter = (imageShape[0] ** 2 + imageShape[1] ** 2) ** 0.5
#     rhoRange = [i for i in range(int(imageDiameter) + 1)]
#     thetaRange = [dTheta * i for i in range(int(-np.pi / (2 * dTheta)), int(np.pi / dTheta))]
#     cosTheta = []
#     sinTheta = []
#     for theta in thetaRange:
#         cosTheta.append(np.cos(theta))
#         sinTheta.append(np.sin(theta))
#     countMatrixSize = (len(rhoRange), len(thetaRange))
#     countMatrix = np.zeros(countMatrixSize)
#
#     eds = []
#     for (x, y), value in np.ndenumerate(edges):
#         if value > 0:
#             eds.append((x, y))
#
#     for thetaIndex in range(len(thetaRange)):
#         theta = thetaRange[thetaIndex]
#         cos = cosTheta[thetaIndex]
#         sin = sinTheta[thetaIndex]
#         for x, y in eds:
#             targetRho = x * cos + y * sin
#             closestRhoIndex = int(round(targetRho))
#             countMatrix[closestRhoIndex, thetaIndex] += 1
#     lines = []
#     for (p, t), value in np.ndenumerate(countMatrix):
#         if value > threshold:
#             lines.append((p, thetaRange[t]))
#     return lines


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
    max_filter = maximum_filter(hough[0], size=7)
    # hough = houghLines(canny, 0.01, 20)

    cv2.imshow("hist", hough[0] / np.max(hough[0]))

    mat = np.repeat(hough[0] / np.max(hough[0]), 3).reshape((*hough[0].shape, 3))
    mat[(max_filter == hough[0]) & (max_filter >= 50), 0] = 0
    mat[(max_filter == hough[0]) & (max_filter >= 50), 1] = 0
    mat[(max_filter == hough[0]) & (max_filter >= 50), 2] = 255

    mat = cv2.resize(mat, (1000, 1000))

    cv2.imshow("hist2", mat)

    lines = np.array(np.where((max_filter == hough[0]) & (max_filter >= 50))).transpose()

    canny = cv2.cvtColor(canny.astype(np.uint8), cv2.COLOR_GRAY2BGR)

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
        cv2.line(canny, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imshow("test", canny)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
