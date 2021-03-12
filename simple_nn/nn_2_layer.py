import numpy as np
import scipy.special


# определение класса нейронной сети двумя скрытими слоями
class NeuralNetwork:

    # инициализация нейронной сети
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # количество узлов во входном, скрытом и выходном слое соответственно
        self.inodes = inputnodes
        self.hnodes = hiddennodes
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
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5),
                                       (self.hnodes, self.inodes))
        # матрица весовых коэффициентов связей между скрытым и выходным слоями
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5),
                                       (self.onodes, self.hnodes))

    # использование сигмоиды в качестве функции активации
    def activation_function(self, x):
        return scipy.special.expit(x)

    # тренировка нейронной сети
    def train(self, inputs_list, targets_list):
        # преобразование списка входных значений в двумерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # рассчет входящих сигналов для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        # рассчет исходящих сигналов для сыкрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # рассчет входящих сигналов для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        # рассчет исходящих сигналов для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # ошибки выходного слоя
        output_errors = targets - final_outputs
        # ошибки скрытогоо слоя - это output_errors, распределенные пропорционально
        # весовым коэффициентам связй и ркомбинированные на скрытых узлах
        hidden_errors = np.dot(self.who.T, output_errors)

        # обновление всовых коэффициентов для связей между скрытым и выходным слоями
        self.who += self.lr * np.dot((output_errors * final_outputs *
                                         (1.0 - final_outputs)),
                                        np.transpose(hidden_outputs))

        # обновление всовых коэффициентов для связеей между входным и скрытым слоями
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs *
                                         (1.0 - hidden_outputs)),
                                        np.transpose(inputs))

    # опрос нейронной сети
    def query(self, inputs_list):
        # преобразование списка входных значений в двумерный массив
        inputs = np.array(inputs_list, ndmin=2).T

        # рассчет входящих сигналов для сыкрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        # рассчет исходящих сигналов для сыкрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # рассчет входящих сигналов для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        # рассчет исходящих сигналов для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs