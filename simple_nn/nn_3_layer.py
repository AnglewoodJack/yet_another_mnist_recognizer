import numpy as np
import scipy.special


# определение класса нейронной сети двумя скрытими слоями
class NeuralNetworkExt:

	# инициализация нейронной сети
	def __init__(self, inputnodes, hiddennodes_1, hiddennodes_2, outputnodes, learningrate):
		# количество узлов во входном, скрытом и выходном слое соответственно
		self.inodes = inputnodes
		self.hnodes1 = hiddennodes_1
		self.hnodes2 = hiddennodes_2
		self.onodes = outputnodes
		# коэффициент обучения
		self.lr = learningrate
		"""
		Для вычисления весов используется нормальное распределение. Центр нормального
		распределения устанавливается в нуле. Стандартное отклонение вычисляется по
		количеству узлов в следующем слое (квадратный коерень из количества узлов).
		Конфигурация массива задается количеством входных/выходных/скрытих узлов.
		"""
		# матрица весовых коэффициентов связей между входным и скрытым слоями
		self.wih = np.random.normal(0.0, pow(self.hnodes1, -0.5),
									   (self.hnodes1, self.inodes))
		# матрица весовых коэффициентов связей между скрытыми слоями
		self.whh = np.random.normal(0.0, pow(self.hnodes2, -0.5),
									   (self.hnodes2, self.hnodes1))
		# матрица весовых коэффициентов связей между скрытым и выходным слоями
		self.who = np.random.normal(0.0, pow(self.onodes, -0.5),
									   (self.onodes, self.hnodes2))
		# использование сигмоиды в качестве функции активации
		self.activation_function = lambda x: scipy.special.expit(x)

	# тренировка нейронной сети
	def train(self, inputs_list, targets_list):
		# преобразование списка входных значений в двумерный массив
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T

		# рассчет входящих сигналов для скрытого слоя 1
		hidden_inputs_1 = np.dot(self.wih, inputs)
		# рассчет исходящих сигналов для сыкрытого слоя 1
		hidden_outputs_1 = self.activation_function(hidden_inputs_1)

		# рассчет входящих сигналов для скрытого слоя 2
		hidden_inputs_2 = np.dot(self.whh, hidden_outputs_1)
		# рассчет исходящих сигналов для сыкрытого слоя 2
		hidden_outputs_2 = self.activation_function(hidden_inputs_2)

		# рассчет входящих сигналов для выходного слоя
		final_inputs = np.dot(self.who, hidden_outputs_2)
		# рассчет исходящих сигналов для выходного слоя
		final_outputs = self.activation_function(final_inputs)

		# ошибки выходного слоя
		output_errors = targets - final_outputs
		# ошибки скрытогоо слоя - это output_errors, распределенные пропорционально
		# весовым коэффициентам связй и ркомбинированные на скрытых узлах
		hidden_errors_2 = np.dot(self.who.T, output_errors)
		hidden_errors_1 = np.dot(self.whh.T, hidden_errors_2)

		# обновление всовых коэффициентов для связей между скрытым и выходным слоями
		self.who += self.lr * np.dot((output_errors * final_outputs *
										 (1.0 - final_outputs)),
										np.transpose(hidden_outputs_2))

		# обновление всовых коэффициентов для связей между скрытыми слоями
		self.whh += self.lr * np.dot((hidden_errors_2 * hidden_outputs_2 *
										 (1.0 - hidden_outputs_2)),
										np.transpose(hidden_outputs_1))

		# обновление всовых коэффициентов для связеей между входным и скрытым слоями
		self.wih += self.lr * np.dot((hidden_errors_1 * hidden_outputs_1 *
										 (1.0 - hidden_outputs_1)),
										np.transpose(inputs))

	# опрос нейронной сети
	def query(self, inputs_list):
		# преобразование списка входных значений в двумерный массив
		inputs = np.array(inputs_list, ndmin=2).T

		# рассчет входящих сигналов для сыкрытого слоя 1
		hidden_inputs_1 = np.dot(self.wih, inputs)
		# рассчет исходящих сигналов для сыкрытого слоя 1
		hidden_outputs_1 = self.activation_function(hidden_inputs_1)

		# рассчет входящих сигналов для сыкрытого слоя 2
		hidden_inputs_2 = np.dot(self.whh, hidden_outputs_1)
		# рассчет исходящих сигналов для сыкрытого слоя 2
		hidden_outputs_2 = self.activation_function(hidden_inputs_2)

		# рассчет входящих сигналов для выходного слоя
		final_inputs = np.dot(self.who, hidden_outputs_2)
		# рассчет исходящих сигналов для выходного слоя
		final_outputs = self.activation_function(final_inputs)

		return final_outputs
