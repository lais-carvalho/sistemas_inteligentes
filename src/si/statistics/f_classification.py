import numpy as np
import scipy
from scipy.stats import f_oneway
from si.data.dataset import Dataset


def f_classification(dataset: Dataset) -> tuple:
    """
    Performs F-test on the dataset to analyze the variance between classes.

    Arguments:
    dataset -- the Dataset object, which should contain samples and labels

    Returns:
    tuple: (F_values, p_values)
    """

    # Group samples by classes
    classes = dataset.get_classes()
    groups = []
    for class_ in classes:
        mask = dataset.y == class_
        group = dataset.x[mask,:]
        groups.append(group)
    return scipy.stats.f_oneway(*groups)
