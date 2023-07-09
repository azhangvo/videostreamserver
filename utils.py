import cv2
import numpy as np


def convolve(mat, kernel):  # Assume mat is 2d grayscale and kernel is square
    kernel_size = kernel.shape[0]
    convolved_mat = np.zeros((mat.shape[0] - kernel_size + 1, mat.shape[1] - kernel_size + 1))

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

    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    convolved_x = convolve(grayscale, kernel)

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
