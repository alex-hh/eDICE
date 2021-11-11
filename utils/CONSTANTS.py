import os
from pathlib import Path


ALL_CHROMOSOMES = ['chr{}'.format(i) for i in list(range(1,23))] + ['chrX']

DATA_DIR = Path(os.environ.get('DATA_DIR', 'localdata'))
CONFIG_DIR = Path(os.environ.get('CONFIG_DIR', 'experiment_config'))
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', 'outputs'))

LOCAL = str(DATA_DIR) == 'data'

basedir = os.path.dirname(os.path.dirname(__file__))
