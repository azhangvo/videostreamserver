import math

import cv2
import numpy as np
from scipy import signal

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


def Canny(mat):
    blurred = Gaussian(mat, kernel_size=5, sigma=1.4)
    gradient, direction = Sobel(blurred)

    # Gradient Magnitude Thresholding

    discretized_direction = np.zeros(direction.shape, dtype=np.uint8)

    # 0 ( -22.5 -  22.5 ) -> left to right
    # 1 (  22.5 -  67.5 ) -> northeast to southwest
    # 2 (  67.5 - 112.5 ) -> north to south
    # 3 ( 112.5 - 157.5 ) -> northwest to southeast
    discretized_direction[(direction > np.pi / 8) & (direction < np.pi * 3 / 8)] = 1
    discretized_direction[(direction > np.pi * 3 / 8) | (direction < -np.pi * 3 / 8)] = 2
    discretized_direction[(direction < -np.pi / 8) & (direction > -np.pi * 3 / 8)] = 3

    # neighbor_directions = np.array([
    #     [[0, 0], [1, -1]],
    #     [[1, -1], [-1, 1]],
    #     [[1, -1], [0, 0]],
    #     [[1, -1], [1, -1]]
    # ], dtype=np.uint8)
    neighbor_directions = np.array([
        [[0, 1], [0, -1]],
        [[-1, 1], [1, -1]],
        [[1, 0], [-1, 0]],
        [[1, 1], [-1, -1]]
    ])

    # gradient_neighbors = np.zeros((*gradient.shape, 2, 2), dtype=np.uint8)

    # print(neighbor_directions[0] + np.array([0, 0]))

    # for i in range(gradient.shape[0]):
    #     for j in range(gradient.shape[1]):
    #         for dir in neighbor_directions[discretized_direction[i][j]]:
    #             ny = i + dir[0]
    #             nx = j + dir[1]
    #             if ny < 0 or nx < 0 or ny >= gradient.shape[0] or nx >= gradient.shape[1]:
    #                 continue
    #             if gradient[i, j] < gradient[ny, nx]:
    #                 gradient[i, j] = 0
    #         # gradient_neighbors[i][j] = neighbor_directions[discretized_direction[i][j]] + np.array([i, j])

    # print(gradient_neighbors.shape)
    #
    # gradient[gradient < gradient[gradient_neighbors[:, :, 0, :]]] = 0
    # gradient[gradient < gradient[gradient_neighbors[:, :, 1, :]]] = 0

    # Please don't look its 2am and my brain wont work

    gradient_neighbors = np.array(list(zip(np.repeat(np.arange(gradient.shape[0]), gradient.shape[1]),
                                           np.tile(np.arange(gradient.shape[1]), gradient.shape[0])))).transpose()

    neighbors1 = gradient_neighbors + neighbor_directions[discretized_direction.flatten()][:, 0, :].transpose()
    neighbors2 = gradient_neighbors + neighbor_directions[discretized_direction.flatten()][:, 1, :].transpose()

    neighbors1[0][neighbors1[0, :] < 0] = 0
    neighbors1[0][neighbors1[0] >= gradient.shape[0]] = gradient.shape[0] - 1
    neighbors1[1][neighbors1[1, :] < 0] = 0
    neighbors1[1][neighbors1[1] >= gradient.shape[1]] = gradient.shape[1] - 1
    neighbors2[0][neighbors2[0, :] < 0] = 0
    neighbors2[0][neighbors2[0] >= gradient.shape[0]] = gradient.shape[0] - 1
    neighbors2[1][neighbors2[1, :] < 0] = 0
    neighbors2[1][neighbors2[1] >= gradient.shape[1]] = gradient.shape[1] - 1

    # gradient = gradient.flatten()
    gradient[gradient < gradient[tuple(neighbors1)].reshape(gradient.shape)] = 0
    gradient[gradient < gradient[tuple(neighbors2)].reshape(gradient.shape)] = 0

    gradient[gradient >= 100] = 255
    gradient[(gradient < 100) & (gradient > 30)] = 150

    edge_strength = np.zeros(gradient.shape, dtype=np.uint8)
    edge_strength[gradient >= 30] = 1
    edge_strength[gradient >= 100] = 2

    return gradient


if __name__ == '__main__':
    image = cv2.imread("./random_photo.jpg")

    cv2.imshow("original", image)

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    cv2.waitKey(0)

    cv2.destroyAllWindows()
