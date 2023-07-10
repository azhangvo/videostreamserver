import math

import cv2
import numpy as np


# 1. reshape image into image cols
# 2. reshape kernel into linear shape
# 3. matrix multiply kernel and image cols
# 4. reshape result into convolved image
def convolve(mat, kernel):
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


def Gaussian(mat, kernel_size=5, sigma=1):
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

    print(gaussian_filter)

    return convolve(mat, gaussian_filter)


def Sobel(mat):
    x_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
    y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)

    convolved_x = convolve(mat, x_kernel)
    convolved_x[convolved_x == 0] = 1e-6
    convolved_y = convolve(mat, y_kernel)

    return np.sqrt(convolved_x ** 2 + convolved_y ** 2), np.arctan(convolved_y / convolved_x)


if __name__ == '__main__':
    image = cv2.imread("./random_photo.jpg")

    cv2.imshow("original", image)

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("grayscale", grayscale)

    print(grayscale)
    print(grayscale.shape)

    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    convolved_x = convolve(grayscale, kernel)
    # naive_convolved_x = naive_convolve(grayscale, kernel)
    #
    # np.testing.assert_array_equal(convolved_x, naive_convolved_x)

    convolved_x[convolved_x < 0] = 0
    convolved_x[convolved_x > 255] = 255
    convolved_x = convolved_x.astype(np.uint8)

    cv2.imshow("convolved x", convolved_x)

    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    convolved_y = convolve(grayscale, kernel)

    convolved_y[convolved_y < 0] = 0
    convolved_y[convolved_y > 255] = 255
    convolved_y = convolved_y.astype(np.uint8)

    cv2.imshow("convolved y", convolved_y)

    gaussian = Gaussian(grayscale)

    print(gaussian)

    cv2.imshow("gaussian", gaussian.astype(np.uint8))

    cv2.waitKey(0)

    cv2.destroyAllWindows()
