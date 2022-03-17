"""A Python package with out of the box torch classes to easily design and train DCNN and MC-DCNN models"""

# Add imports here
from .eztorch4conv import *
import torch

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
