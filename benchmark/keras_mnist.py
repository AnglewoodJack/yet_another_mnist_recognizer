from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.datasets import mnist

# Загрузка набора данных MNIST
from preprocess.deskew import apply_deskew

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Архитектура сети
network = models.Sequential()
network.add(layers.Dense(800, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# Компиляция
network.compile(optimizer='rmsprop',
				loss='categorical_crossentropy',
				metrics=['accuracy'])

# Подготовка исходных данных
train_images = apply_deskew(train_images)
test_images = apply_deskew(test_images)

# Подготовка меток
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Обучение
network.fit(train_images, train_labels, epochs=20, batch_size=32)

# Предсказание
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('accuracy: ', test_acc)
