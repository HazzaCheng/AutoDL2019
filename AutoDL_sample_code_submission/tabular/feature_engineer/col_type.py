import numpy as np
from collections import Counter
from datetime import datetime
from enum import Enum, unique


@unique
class ColType(Enum):
    Num = 0
    Cat = 1
    Time = 2


def infer_type(x):
    col_types = {}
    sample_count = len(x)
    time_lower_limit = datetime(1980, 1, 1).timestamp()
    time_upper_limit = datetime(2021, 1, 1).timestamp()
    judge_time_col = np.vectorize(
        lambda val: not (time_lower_limit <= val <= time_upper_limit)
    )
    for i in range(len(x[0])):
        is_not_time = sum(
            judge_time_col(x[:, i])
        )
        if is_not_time == 0:
            col_types[i] = ColType.Time
            continue

        counter = Counter(x[:, i])
        value_counts = len(counter.keys())

        if value_counts < sample_count * 0.01:
            is_not_digit = sum(np.vectorize(lambda x: not np.isnan(x) and int(x) != x)(x[:, i]))
            if is_not_digit == 0:
                col_types[i] = ColType.Cat
            else:
                col_types[i] = ColType.Num
        else:
            col_types[i] = ColType.Num
    return col_types
