from keras import models
from keras import layers
from keras import optimizers
from keras import callbacks
from keras.utils import to_categorical
from keras.datasets import mnist

from pathlib import Path

# Загрузка набора данных MNIST
from preprocess.deskew import apply_deskew
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Подготовка исходных данных
train_images = apply_deskew(train_images).reshape((60_000, 28, 28, 1))
test_images = apply_deskew(test_images).reshape((10_000, 28, 28, 1))

# Подготовка меток
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Путь к файлу обученной модели
path = Path.cwd() / "conv_trained"
path.mkdir(exist_ok=True, parents=True)
assert path.exists()
cpt_filename = "best_checkpoint.hdf5"
cpt_path = str(path / cpt_filename)

# Архитектура сети
model = models.Sequential()
# Добавление слоёв свёрточной нейронной сети.
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Добавление классификатора поверх свёрточной нейронной сети.
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Оптимизация градиентного спуска
optimizer = optimizers.Adam(lr=0.001)

# Компиляция
model.compile(optimizer=optimizer,
				loss='categorical_crossentropy',
				metrics=['accuracy'])

# запись чекпоинтов
checkpoint = callbacks.ModelCheckpoint(cpt_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Обучение
model.fit(train_images, train_labels, validation_data=(test_images, test_labels),
			epochs=20, batch_size=32, verbose=1, callbacks=[checkpoint])
