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

    cv2.waitKey(0)

    cv2.destroyAllWindows()
