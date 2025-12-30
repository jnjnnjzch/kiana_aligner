# kiana/__init__.py

import sys
# 让用户可以直接通过 from kiana import ... 来使用核心类
from .loaders import MatLoader, DataFrameLoader, TrcLoader, SeqLoader
from .ephys import EphysProcessor
from .behavior import BehavioralProcessor
from .analysis import SpikeTrainAnalyzer

# 定义版本号，这是一个好习惯
if sys.version_info >= (3, 8):
    from importlib.metadata import version, PackageNotFoundError
else:
    # 如果用户还在使用Python 3.7，则使用后备库
    from importlib_metadata import version, PackageNotFoundError

try:
    # `__package__`会自动获取当前包的名称(即"kiana")
    # `version()`函数会查找已安装的、名为"kiana"的包的版本信息
    __version__ = version(__package__ or "kiana")
except PackageNotFoundError:
    # 如果包没有通过pip安装（例如，只是在本地作为脚本运行），
    # `version()`会找不到元数据并报错。我们捕获这个错误，
    # 并给出一个默认的开发版本号。
    __version__ = "0.0.0-dev"

print(f"KIANA Toolkit v{__version__} initialized.")