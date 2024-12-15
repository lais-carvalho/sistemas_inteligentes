from si.data.dataset import Dataset
import numpy as np

def train_test_split(dataset: Dataset, test_size: float, random_state=123):
    np.random.seed(random_state)

    permutations = np.random.permutation(dataset.X.shape()[0])

    test_sample_size = int(dataset.shape()[0]*test_size)

    test_idx = permutations[:test_sample_size]
    train_idx = permutations[test_sample_size:]

    train_dataset = Dataset(dataset.X[train_idx, :], dataset.y[train_idx, :])



    num_samples = len(dataset)
    num_test_samples = int(test_size * num_samples)
    num_train_samples = num_samples - num_test_samples

    train_indices = permutations[:num_train_samples]
    test_indices = permutations[num_train_samples:]

    train_dataset = dataset[train_indices]
    test_dataset = dataset[test_indices]

    return train_dataset, test_dataset