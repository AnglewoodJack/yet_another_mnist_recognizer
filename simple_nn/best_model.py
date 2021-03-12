import pickle

from simple_nn.nn_2_layer import NeuralNetwork

from simple_nn.FitPred import fit, predict, result


# загрузка файла с датасетом рукописных чисел - тренировочный датасет
with open('../mnist_dataset/mnist_train.csv', 'r') as file:
	data_train = file.readlines()

# загрудка тестового набора данных
with open('../mnist_dataset/mnist_test.csv', 'r') as file:
	data_test = file.readlines()


# количество входных, скрытых и выходных узлов
input_nodes = 784
hidden_nodes = 800
output_nodes = 10
# коэффициент обучения
learning_rate = 0.1
# экземпляр нейронной сети
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# обучение нейросети
fit(epochs=7, network=n, train=data_train, out_nodes=output_nodes, deskew_dat=True)
# прогнозирование
scorecard = predict(network=n, test=data_test, deskew_dat=True)
# вывод результата
result(scorecard)
# сохраненние модели в pickle
with open('model.pickle', 'wb') as handle:
	pickle.dump(n, handle)
