import pickle
import tensorflow as tf

from advanced.layer import FCLayer, ReLU
from advanced.network import Network
from preprocess.deskew import apply_deskew

# загрузка датасета MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# преобразование изображений
train_images = apply_deskew(train_images)
test_images = apply_deskew(test_images)

# создание нейросети
layers = [] # список слоев
layers.append(FCLayer(train_images.shape[1], 784)) # входной слой
layers.append(ReLU()) # активация
layers.append(FCLayer(784, 800)) # скрытый слой
layers.append(ReLU()) # активация
layers.append(FCLayer(800, 10)) # выходной слой - 10 классов (10 цифр)

# инициализация нейросети указанными слоями
net = Network(layers=layers)
# обучение и проверка
net.fit(x_train=train_images, y_train=train_labels,
		x_test=test_images, y_test=test_labels,
		batch_size=32, num_epochs=20, learning_rate=0.1)

# сохраненние модели в pickle
with open('adv_model.pickle', 'wb') as handle:
	pickle.dump(net, handle)