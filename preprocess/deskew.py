import numpy as np
from scipy.ndimage import interpolation


# нахождеение центра масс изображения
def moments(image):
    # создание сетки
    c0, c1 = np.mgrid[:image.shape[0], :image.shape[1]]
    # сумма пикселей
    total_image = np.sum(image)
    # mu_x - момент по x
    m0 = np.sum(c0 * image) / total_image
    # mu_y - момент по y
    m1 = np.sum(c1 * image) / total_image
    # var(x) - вариация по x
    m00 = np.sum((c0 - m0)**2 * image) / total_image
    # var(y) - вариация по y
    m11 = np.sum((c1 - m1)**2 * image) / total_image
    # covariance(x,y) - ковариация
    m01 = np.sum((c0 - m0) * (c1 - m1) * image) / total_image
    # вектор центов масс - mu_x, mu_y соответственно
    mu_vector = np.array([m0, m1])
    # наблюдается ли схожесть в ковариационных матрицах
    covariance_matrix = np.array([[m00,m01],[m01,m11]])

    return mu_vector, covariance_matrix


# выравнивание
def deskew(image):
    # нахождение моментов
    c, v = moments(image)
    # нахождение коэффициента альфа для матрицы выравнивания
    alpha = v[0, 1] / v[0, 0]
    # матрица выравнивания
    affine = np.array([[1, 0], [alpha, 1]])
    # новый центр масс
    ocenter = np.array(image.shape)/2.0
    # центрирование
    offset = c - np.dot(affine, ocenter)
    # трансформация
    img = interpolation.affine_transform(image, affine, offset=offset)
    return (img - img.min()) / (img.max() - img.min())


# для применени к датасету MNIST из библиотеки TensorFlow
def apply_deskew(data):
    """
	Применение метда deskew (выравивание цифр) ко входным данным.
	:param data: входные данные.
	:return: преобразованные данные.
	"""
    # пустой тензор результатов
    result = np.zeros(data.shape)
    # цикл по каждой записи (картинке с цифрой)
    for i in range(data.shape[0]):
        image = data[i,...].astype('float32') / 255.
        # применение deskew
        processed_image = deskew(image)
        # хапись результатов
        result[i] = processed_image

    return result.reshape(result.shape[0], -1)
