import numpy as np


def multiclass_crossentropy_with_logits(logits, y_true):
	"""
	Функция расчета потерь.
	:param logits: выход нейронной сети без активации. Размерность: (N, k),
	где N -количество объектов, k -количество классов.
	:param y_true: реальные классы для N объектов.
	:return вектор: потерь на каждом объекте.
	"""
	logits_for_answers = logits[np.arange(len(logits)), y_true]
	cross_entropy = -logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))

	return cross_entropy


def grad_multiclass_crossentropy_with_logits(logits, y_true):
	"""
	Функция расчета производных потерь.
	:param logits: выход нейронной сети без активации. Размерность: (N, k),
	где N - количество объектов, k - количество классов.
	:param y_true: реальные классы для N объектов.
	:return: возвращает матрицу производных.
	"""
	ones_for_answers = np.zeros_like(logits)
	ones_for_answers[np.arange(len(logits)), y_true] = 1

	softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

	return (- ones_for_answers + softmax) / logits.shape[0]
