from typing import Callable, Tuple, Dict, Any
import itertools

import numpy as np

from src.si.data.dataset import Dataset
from src.si.model_selection.cross_validate import k_fold_cross_validation


def randomized_search_cv(model,
                   dataset: Dataset,
                   hyperparameter_grid: Dict[str, Tuple],
                   scoring: Callable = None,
                   cv: int = 5, n_iter: int = 10) -> Dict[str, Any]:

    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")

    results = {'scores': [], 'hyperparameters': []}

    all_combinations = list(itertools.product(*hyperparameter_grid.values()))
    if n_iter >= len(all_combinations):
        raise ValueError("n_iter cannot exceed the total number of hyperparameter combinations.")

    random_indices = np.random.choice(len(all_combinations), n_iter, replace=False)
    selected_combinations = [all_combinations[i] for i in random_indices]

    for combination in selected_combinations:
        # parameter configuration
        parameters = {}

        # set the parameters
        for parameter, value in zip(hyperparameter_grid.keys(), combination):
            setattr(model, parameter, value)
            parameters[parameter] = value

        # cross validate the model
        score = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # add the score
        results['scores'].append(np.mean(score))

        # add the hyperparameters
        results['hyperparameters'].append(parameters)

    results['best_hyperparameters'] = results['hyperparameters'][np.argmax(results['scores'])]
    results['best_score'] = np.max(results['scores'])
    return results
