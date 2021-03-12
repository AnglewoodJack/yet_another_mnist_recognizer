# yet_another_mnist_recognizer
Распознавание  рукописных цифр из датасета MNIST с помощью простейшей нейросети.
## Описание

Реализовано распознавание рукописных цифр на наборе данных [MNIST](http://yann.lecun.com/exdb/mnist/) с помощью базовой<br>
нейронной сети, созданной без использования специализированных библиотек:
<br><br>
1. На основе двухслойной нейросети (скрытый и выходной слои), подробно описанной в книге
   <br>
   "Make Your Own Neural Network" (Tariq Rashid). [Ссылка](https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork) на соответсвующий
   рпозитрий автора.
   <br><br>
   Тренировочный и тестовый датасеты взяты с [сайта автора](http://makeyourownneuralnetwork.blogspot.com/2015/03/the-mnist-dataset-of-handwitten-digits.html) в формате .csv.
   <br><br>
   В качестве препроцессинга данных использовано "выравнивание" (deskew) - [источник](https://fsix.github.io/mnist/Deskewing.html)
   <br> - с некоторыми [модификациями](https://stackoverflow.com/questions/43577665/deskew-mnist-images).
   <br><br>
   Лучший результат на тестовом датасете: `accuracy = 0.9838 (error = 1.62 %)`
   <br><br>
2. На основе нейросети из курса «Профессия Data Scientist‌» от онлайнн унниверситета Skillbox.
   <br><br>
   Тренировочный и тестовый датасеты загружены из библиотеки TensorFlow.<br>
   В качестве препроцессинга также использован deskew.
   <br><br>
   Лучший результат на тестовом датасете: `accuracy = 0.9873 (error = 1.27 %)`
   

## Состав репозитория
Папка `simple_nn`:<br>
- `nn_2_layer.py` - двухслойная нейросеть (один скрытий и выходной слой)<br>
- `nn_3_layer.py` - трехслойная нейросеть (два скрытых и один выходной слой)<br>
- `FitPred.py` - функции для тренировки и опроса нейросети<br>
- `best_model.py` - скрипт с реализацией лучшей модел из ноутбука "research.ipynb"<br>
- `model.pickle` - обученная модель с лучшими показателями<br>

Папка `preprocess`:<br>
- `layer.py` - определение слоев нейросети
- `loss.py` - функции для нахожденния потерь
- `batch.py` - раазбиение на минибатчи
- `network.py` - создание нейросети
- `best_model.py` - скрипт с реализацией лучшей модели<br>
- `model.pickle` - обученная модель с лучшими показателями<br>

Папка `preprocess`:<br>
- `deskew.py` - функции для прероцессинга данных - "выравнивание"<br>
- `elastic.py` - функции для прероцессинга данных - упругое искажение<br>

`research.ipynb` - ноутбук с подборов наилучших параметров модели<br>
`requirements.txt` - зависимости<br>