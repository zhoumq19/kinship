import os
from time import strftime, localtime
from typing import Tuple, List, Dict

from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


def mkdir_chdir(directory: str):
    os.makedirs(directory, exist_ok=True)
    os.chdir(directory)


def get_int_dtype(num: int):
    if - 2 ** 8 <= num < 2 ** 8:
        return np.int8
    if - 2 ** 16 <= num < 2 ** 16:
        return np.int16
    if - 2 ** 32 <= num < 2 ** 32:
        return np.int32
    # if - 2 ** 64 <= num < 2 ** 64:
    #     return np.int64
    return np.int64


def get_uint_dtype(num: int):
    if num < 2 ** 8:
        return np.uint8
    if num < 2 ** 16:
        return np.uint16
    if num < 2 ** 32:
        return np.uint32
    # if num < 2 ** 64:
    #     return np.uint64
    return np.uint64


def print2(*args, file: str = 'log.txt', mode: str = 'a',
           add_time: bool = True, print2console: bool = True,
           sep='\t', end='\n') -> None:
    if add_time:
        args = (strftime(f'%Y-%m-%d{sep}%H:%M:%S', localtime()), *args)
    if print2console:
        print(*args, sep=sep, end=end)
    with open(file, mode=mode, encoding='utf-8') as f:
        print(*args, sep=sep, end=end, file=f)


def print_error(*args, file: str = 'contradiction.txt', mode: str = 'a',
                sep='\t', end='\n') -> None:
    args = (strftime(f'%Y-%m-%d{sep}%H:%M:%S', localtime()), *args)
    print(*args, sep=sep, end=end)
    with open(file, mode=mode, encoding='utf-8') as f:
        print(*args, sep=sep, end=end, file=f)


def print_df(df, file: str = 'node operation.txt', mode: str = 'a', sep='\t', end='\n') -> None:
    with open(file, mode=mode, encoding='utf-8') as f:
        for row in df:
            print(*row, sep=sep, end=end, file=f)


def ngrams(sequence, n: int):
    """
    from nltk import ngrams
    Examples
    --------
    >>> seq = 'abcdefg'
    >>> list(ngrams(seq, n=2))
    ['ab', 'bc', 'cd', 'de', 'ef', 'fg']
    """
    sequence_len = len(sequence)
    for i in range(sequence_len - n + 1):
        yield sequence[i:(i + n)]
