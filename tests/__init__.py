import os

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data/processed/data.pt")  # root of data
_PATH_CONF = os.path.join(_PROJECT_ROOT, "conf/config.yaml")  # path of config
_PATH_CKPT = os.path.join(_PROJECT_ROOT, "models/checkpoint.ckpt")  # path of config
