import numpy as np
from src.si.data.dataset import Dataset
from typing import Tuple

def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    np.random.seed(random_state)
    unique_classes, class_counts = np.unique(dataset.y, return_counts=True)
    idxs_train =[]
    idxs_test = []
    for class_l, class_count in zip(unique_classes, class_counts):
        n_test_samples = int(class_count * test_size)

        class_idxs = np.where(dataset.y == class_l)[0]
        np.random.shuffle(class_idxs)

        test_class_indices = class_idxs[:n_test_samples]

        train_class_indices = class_idxs[n_test_samples:]

        idxs_test.extend(test_class_indices)
        idxs_train.extend(train_class_indices)

    train = Dataset(dataset.X[idxs_train], dataset.y[idxs_train], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[idxs_test], dataset.y[idxs_test], features=dataset.features, label=dataset.label)
    return train, test