"""
A Python package with out of the box torch classes to easily design and train 
DCNN and MC-DCNN models
"""

# Add imports here
from .callbacks import Callback, early_stop, checkpoint
from .architectures import dcnn, trainer, mcdcnn
from .layers import conv3d, dense, fire3d
from .utils import parse_inputs

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
