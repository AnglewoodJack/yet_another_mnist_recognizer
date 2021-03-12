import numpy as np


class Layer:
	"""
	Базовый класс слоя нейронной сети.
	Все слои наследуются от данного класса и реализуют два метода: forward и backward.
	"""
	def forward(self, x):
		pass
	def backward(self, dL_dz, learning_rate=0):
		pass


class ReLU(Layer):
	"""
	Слой с функцией активации ReLU.
	"""
	def forward(self, x):
		"""
		Прямой проход.
		Метод для вычисления ReLU(x).
		:param x: входной сигнал слоя.
		:return: выходной сигнал, после применения ReLU на слое.
		"""
		# сохраняем вход
		self._saved_input = x
		# выход ReLU
		output = np.maximum(x,0)
		# размерность выхода должна совпадать с размерностью входа
		assert output.shape == x.shape
		return output

	def backward(self, dL_dz, learning_rate=0.):
		"""
		Обратный проход.
		Нахождение производной dL_dx = dL_dz * dz_dx (в соответствии с chain rule).
		Для слоя relu, dz_dx(x) = 1, при x > 0, и dz_dz = 0 при x < 0.
		:param dL_dz: производная финальной функции по выходу этого слоя.
		:param learning_rate: не используется, т.к. ReLU не содержит параметров.
		:return: выходной сигнал, после обратного прохода.
		"""
		# производная выхода ReLU по ее входу
		dz_dx = (self._saved_input > 0.0).astype(np.int32)
		# размерость должна в точности соответствовать размерности "x" из forward pass
		assert dz_dx.shape == self._saved_input.shape,\
			f"Shapes must be the same. Got {dz_dx.shape, self._saved_input.shape}"
		# выход
		output = dz_dx * dL_dz
		return output


class FCLayer(Layer):
	"""
	Полносвязный (fully connected/dense) слой.
	"""
	def __init__(self, in_dim, out_dim):
		"""
		:param in_dim: количество входных нейронов.
		:param out_dim: количество выходных нейронов.
		"""
		self.in_dim = in_dim
		self.out_dim = out_dim
		# инициализация матрицы весов (in_dim, out_dim) с помощью нормального распределения
		self.weights = np.random.randn(in_dim, out_dim) * 0.001
		# инициализация смещения нулями
		self.bias = np.zeros(self.out_dim)
		self._saved_input = None

	def forward(self, x: np.ndarray) -> np.ndarray:
		"""
		Прямой проход.
		Вычисление выхода полносвязного слоя.
		:param x: вход слоя, размерности (N, in_dim), где N - количество объектов в батче
		:return: matmul(x, weights) + bias
		"""
		# проверка размерностей
		assert np.ndim(x) == 2, f"Inputs dimension must be 2. Got {np.ndim(x)}"
		assert x.shape[1] == self.in_dim, f"Inputs shape must correspond to in_dim. Got {x.shape[1]}"
		# сохранение входного сигнала
		self._saved_input = x
		# выход полносвязного слоя
		output = np.dot(self._saved_input, self.weights) + self.bias
		# проверка размерностей
		assert output.shape == (x.shape[0], self.out_dim),\
			f"Shapes do not match {(output.shape, (x.shape[0], self.out_dim))}"
		return output

	def backward(self, dL_dz, learning_rate=0.):
		"""
		Обратный проход.
		Нахождение производной dL_dx.
		:param dL_dz: производная финальной функции по выходу этого слоя. Размерость (N, self.out_dim).
		:param learning_rate: скорость обучеиня; если отличен от нуля,то параметры слоя (weights, bias) обновляются.
		:return: выходной сигнал, после обратного прохода.
		"""
		# проверка размерностей
		assert np.ndim(dL_dz) == 2, f"Outputs dimension must be 2. Got {np.ndim(dL_dz)}"
		assert dL_dz.shape[1] == self.out_dim, f"Outputs shape must correspond to out_dim. Got {dL_dz.shape[1]}"
		# скалярное произведение
		self.dL_dw = np.dot(self._saved_input.T, dL_dz)
		self.dL_dx = np.dot(dL_dz, self.weights.T)
		# смещение
		self.dL_db = dL_dz.sum(0)

		assert self.dL_db.shape == self.bias.shape,\
			f"Bias shape mismatch. Got {self.dL_db.shape, self.bias.shape}"
		assert self.dL_dw.shape == self.weights.shape,\
			f"Weights shape mismatch. Got {self.dL_dw.shape, self.weights.shape}"
		assert self.dL_dx.shape == self._saved_input.shape,\
			f"Inputs shape mismatch. Got {self.dL_dx.shape, self._saved_input.shape}"

		if learning_rate != 0:
			# шаг градиентного спуска
			self.weights -= learning_rate * self.dL_dw
			self.bias -= learning_rate * self.dL_db

		return self.dL_dx
