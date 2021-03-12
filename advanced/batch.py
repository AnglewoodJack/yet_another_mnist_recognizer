import numpy as np

from tqdm import tqdm

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	"""
	Разделение датасета на минибатчи.
	:param inputs: датасет.
	:param targets: метки датасета.
	:param batchsize: размер батча.
	:param shuffle: перемешивание.
	:return: датасет + метки датасета, соотвествующие размеру батча.
	"""
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.random.permutation(len(inputs))
	for start_idx in tqdm(range(0, len(inputs) - batchsize + 1, batchsize)):
		if shuffle:
			batch_indexes = indices[start_idx:start_idx + batchsize]
		else:
			batch_indexes = slice(start_idx, start_idx + batchsize)

		yield inputs[batch_indexes], targets[batch_indexes]
