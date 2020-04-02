import numpy as np
from sklearn.model_selection import train_test_split
from hyperopt import STATUS_OK, Trials, space_eval, tpe, fmin

from tools import log


def bayes_opt(X, y, params, space, classifier_class, metric, is_larger_better=True, **kwargs):
    """
    Use bayesian opt to search hyper parameters
    :param X: dataset
    :param y: labels
    :param params: fix params
    :param space: hyper parameters search space
    :param classifier_class: model class
    :param metric: function to calculate metric, should be func(y_true, y_pred, *args, **kwargs)
    :param is_larger_better: whether the larger the metric, the better
    :return: best hyper parameters
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

    def objective(hyperparams):
        all_params = {**params, **hyperparams}
        log("params:\n{}".format(all_params))
        classifier = classifier_class(**all_params)
        classifier.fit((X_train, y_train))
        y_pred = classifier.predict(X_val)
        y_val_ohe = np.eye(all_params["num_class"])[y_val]
        score = metric(y_val_ohe, y_pred)
        if is_larger_better:
            score = -score
        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()
    log("kwargs:\n{}".format(kwargs))
    kwargs['max_evals'] = 50 if 'max_evals' not in kwargs else kwargs['max_evals']
    best = fmin(
        fn=objective,
        space=space,
        trials=trials,
        algo=tpe.suggest,
        verbose=1,
        rstate=np.random.RandomState(66),
        show_progressbar=False,
        **kwargs
    )

    hyperparams = space_eval(space, best)

    log("{} = {:0.4f} {}".format(
        metric.__name__,
        -trials.best_trial['result']['loss'],
        hyperparams
    ))

    return hyperparams
