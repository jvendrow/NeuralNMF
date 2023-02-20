"""Top-level package for Neural NMF."""

# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.0.5"

__author__ = "Joshua Vendrow"
__email__ = "jvendrow@ucla.edu"

__license__ = "MIT"
__copyright__ = "Copyright (c) 2020, Joshua Vendrow"

from .writer import *
from .news_group_loading import *

from .loss import *
from .lsqnonneg_module import *
from .model import *
from .train import *
