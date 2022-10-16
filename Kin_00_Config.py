
import platform
import sys
import os
import re
import json
from os.path import join
from collections import Counter
from typing import List, Set, Dict, Tuple
# import copy
# import argparse

from contextlib import closing
import sqlite3
import pickle
import sklearn
import gensim

from tqdm import tqdm
import numpy as np
import pandas as pd
import regex


from kinship_py.KinshipCode import ComplexKinshipCode, is_compatible
from kinship_py.KinNetwork import KinNetwork
from networkx import all_simple_edge_paths, shortest_path
from networkx import MultiDiGraph, weakly_connected_components, get_edge_attributes

from kinship_py.utils import mkdir_chdir, print2
# def project_dir(project_dir = '/Users/francis/PycharmProjects/Kinship_py'):
#     work_dir = os.path.join(project_dir, 'example_CBDB')
#     data_dir = os.path.join(work_dir, 'data')
#     output_dir = os.path.join(work_dir, 'output')
#     return work_dir, data_dir, output_dir


# wang_case_data_dir = os.path.join(work_dir, 'wang_case/data')
