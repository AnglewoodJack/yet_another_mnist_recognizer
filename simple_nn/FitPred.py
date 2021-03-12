import numpy as np
import time

from tqdm import tqdm
from random import shuffle

from preprocess.deskew import deskew
from preprocess.elastic import elastic_transform


def fit(epochs, network, train, out_nodes, deskew_dat=False, elastic=False):
    """
    Обучение нейросети.
    :param epochs: количество эпох обучения
    :param network: экземпляр класса нейронной сети
    :param train: тренировочныфй датасет
    :param out_nodes: количество выходных нейронов
    :param deskew_dat: препроцессинг данных - выравнивание цифр датасета
    :param elastic: препроцессинг данных - применение упругого искажения, Tuple(alpha, sigma)
    """
    print("Обучение...")

    for epoch in range(epochs):
        # перетасовка входных данных
        shuffle(train)
        # добавить progressbar
        pbar = tqdm(train)
        # перебор всех записей в тренировочном наборе данных
        for record in pbar:
            pbar.set_description(f"Эпоха {epoch+1}")
            # получение списка значний
            all_values = record.split(',')
            # масштабирование входных значений
            _inputs = np.asfarray(all_values[1:]) / 255.0
            # преобразовать в матрицу 28х28
            _temp = _inputs.reshape((28, 28))

            # упругое искажение
            if elastic:
                # применить трансформирование
                _temp_proc = elastic_transform(_temp, elastic[0], elastic[1])
                # преобразовать обратно в вектор из 784 занчений
                _inputs = _temp_proc.flatten()

            # выравнивание цифр на изображении
            if deskew_dat:
                # применить выравнивание
                _temp_proc = deskew(_temp)
                # преобразовать обратно в вектор из 784 занчений
                _inputs = _temp_proc.flatten()
            """
            Перевод значения цветовых кодов из большого диапазона значений 0-255 в меньший
            диапазон от 0.01 до 1.0. Нижнее знначение диапазона выбрано таким, чтобы избежать
            проблем с нулевыми входными значениями - они могут искуственно блокировать обновление
            весов. Для входных сигналов нет необходимости выбирать верхнее значенние диапазона
            равным 0.99, т.к. нет нужды избегать значений 1.0 для входных сигналов. Лишь выходные
            сигналы не могут превышать значение 1.0.
            """
            # смещение входных значений
            inputs = (_inputs * 0.99) + 0.01
            # создани целевых выходных знаачений (все равны 0.01,
            # кроме маркерного занчения, равного 0.99
            targets = np.zeros(out_nodes) + 0.01
            # all_values[0] -  целевое маркерное значение для данной записи
            targets[int(all_values[0])] = 0.99
            network.train(inputs, targets)


def predict(network, test, deskew_dat=False, elastic=False):
    """
    Классификация чисел нейросетью.
    :param network: экземпляр класса нейронной сети.
    :param test: тренировочныфй датасет.
    :param deskew_dat: препроцессинг данных - выравнивание цифр датасета
    :param elastic: препроцессинг данных - применение упругого искажения, Tuple(alpha, sigma)
    :return: список из 0 и 1, 0 - неправильно классифицировано, 1 - правильно.
    """
    print("Клсссификация...")
    time.sleep(1)
    # журнал оценок работы сети, изначально путой
    scorecard = []

    # перебор всех записей в тестовом наборе данных
    for record in tqdm(test, desc="Обработка тестовых данных: "):
        # получение списка значений из записи
        all_values_test = record.split(',')
        # правильный ответ - первое значениее
        correct_label = int(all_values_test[0])
        # масштабирование входных значений
        _inputs = np.asfarray(all_values_test[1:]) / 255.0
        # преобразовать в матрицу 28х28
        _temp = _inputs.reshape((28, 28))

        # упругое искажение
        if elastic:
            # применить трансформирование
            _temp_proc = elastic_transform(_temp, elastic[0], elastic[1])
            # преобразовать обратно в вектор из 784 занчений
            _inputs = _temp_proc.flatten()

        # выравнивание цифр на изображении
        if deskew_dat:
            # применить выравнивание
            _temp_proc = deskew(_temp)
            # преобразовать обратно в вектор из 784 занчений
            _inputs = _temp_proc.flatten()

        # смещение входных значений
        inputs = (_inputs * 0.99) + 0.01
        # опрос сети
        outputs = network.query(inputs)
        # индекс наибольшего значения является маркерным значением
        label = np.argmax(outputs)
        # добавление оценки ответа сети к концу списка
        if (label == correct_label):
            # добаавлние 1 в случае правильного ответа
             scorecard.append(1)
        else:
            # добаавлние 0 в случае неправильного ответа
            scorecard.append(0)

    return scorecard


def result(scorecard):
    # расчет показателя эффективности в виде доли правильных ответов
    scorecard_array = np.asarray(scorecard)
    # эффективность и ошибка
    efficiency = scorecard_array.sum() / scorecard_array.size
    error = np.round((1 - efficiency) * 100, 2)
    print(f"эффективность (ошибка): {efficiency} ({error}%)", )
    return efficiency
