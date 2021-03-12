import numpy as np
import matplotlib.pyplot as plt

from typing import List
from sklearn.metrics import accuracy_score

from advanced.batch import iterate_minibatches
from advanced.layer import Layer
from advanced.loss import multiclass_crossentropy_with_logits, grad_multiclass_crossentropy_with_logits


class Network:
	"""
	Нейронная сеть
	"""
	def __init__(self, layers: List[Layer]):
		"""
		Для инициализации нейронной сети,необходим список слоев, которые должны
		быть последовательно применены друг к другу.
		:param layers: список слоев.
		"""
		self.layers = layers

	def forward(self, x: np.ndarray):
		"""
		Проброс сигнала вперед черех все слои.
		Получив x на вход, сеть должна по-очереди применить к нему все слои.
		Т.е. выход каждого слоя является входом следующего.
		:param x: входной батч объектов размера (N, размер_входа_первого_слоя).
		:return output: выходной сигнал после прохода черех все слои нейронной сети.
		"""
		# последовательное применение forward методов каждого из слоев (self.layers)
		for layer in self.layers:
			x = layer.forward(x)

		output = x

		return output

	def predict(self, x):
		"""
		Фенкция нахождения вероятностей по логитам выходых сигналов последнего слоя.
		:param x: входной батч объектов размера (N, размер_входа_первого_слоя).
		:return: вектор размера (N) с номером предсказанного класса для каждого объекта.
		"""
		# подсчет логитов посредством полного форвард пасса сети
		logits = self.forward(x) # размер логитов (N, k), где k -количество классов
		# получение классов из логитов
		classes = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
		classes = np.argmax(classes.T, axis=0)
		# проверка размерностей
		assert classes.shape == (x.shape[0],), f"Classes shape mismatch {classes.shape, (x.shape[0],)}"
		return classes

	def train_step(self, x, y, learning_rate):
		"""
		Шаг обучения.
		:param x: входной батч объектов размера (N, размер_входа_первого_слоя)
		:param y: реальные классы объектов (N,)
		:param learning_rate: скорость обучения.
		:return:
		"""
		# получение логитов
		logits = self.forward(x)
		# кросс-энтропия логитов
		loss = multiclass_crossentropy_with_logits(logits, y)
		# градиент кросс-энтропии
		loss_grad = grad_multiclass_crossentropy_with_logits(logits, y)
		# проброс loss_grad через всю сеть вызовом backward каждого слоя в обратном порядке
		for layer in self.layers[::-1]:
			loss_grad = layer.backward(loss_grad, learning_rate)
		# усреднение потерь
		return np.mean(loss)

	def fit(self, x_train, y_train, x_test, y_test, learning_rate, num_epochs,
			batch_size):
		"""
		Цикл обучения:
		Итерирование по минибатчам и вызыв на каждом из них train_step;
		Логирование лосса, точности и отрисовка графика.
		:param x_train: тренировочный датасет.
		:param y_train: метки тренировочного датасета.
		:param x_test: тестовый датасет.
		:param y_test: метки тестового датасета.
		:param learning_rate: скорость обучения.
		:param num_epochs: колоичество эпох обучения.
		:param batch_size: размер батча.
		"""
		# пустые списки для записи логов
		train_log = []
		test_log = []
		loss_log = []
		# итерирование по эпохам обучения
		for epoch in range(num_epochs):
			# запись потерь - loss-function
			loss_iters = []
			# разбиение на батчи
			for x_batch,y_batch in iterate_minibatches(x_train, y_train, batchsize=batch_size, shuffle=True):
				# выполнение шага обучения по батчу
				loss_iters.append(self.train_step(x_batch, y_batch, learning_rate=learning_rate))
			# для визуализации усредняем лосс за каждую итерацию
			loss_log.append(np.mean(loss_iters))
			# нахождение точности на тренировочном датасете
			train_accuracy = accuracy_score(y_train, self.predict(x_train))
			# нахождение точности на тестовом датасете
			test_accuracy = accuracy_score(y_test, self.predict(x_test))
			# логирование
			train_log.append(train_accuracy)
			test_log.append(test_accuracy)
			# вывод результатов в консоль
			print("Epoch", epoch)
			print("Train accuracy:",train_log[-1])
			print("Test accuracy:",test_log[-1])
			# отрисовка результатов
			plt.figure(figsize=(10, 5))
			ax1 = plt.subplot(1,2,1)
			plt.plot(train_log,label='train accuracy')
			plt.plot(test_log,label='test accuracy')
			ax2 = plt.subplot(1,2,2)
			plt.plot(loss_log,label='loss')
			ax1.legend(loc='best')
			ax2.legend(loc='best')
			plt.grid()
			plt.tight_layout()
			plt.show()
