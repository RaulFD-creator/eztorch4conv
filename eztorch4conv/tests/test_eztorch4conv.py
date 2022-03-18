"""
Unit and regression test for the eztorch4conv package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import eztorch4conv
from eztorch4conv import *

def test_eztorch4conv_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "eztorch4conv" in sys.modules
    model = DCNN("model1", "./")
    model2 = MCDCNN(3,"model2", "./")
    model.add_layer(conv3d(1,2,3))
    model.add_layer(flatten())
    model.add_layer(dense(40, 2))
    model2.add_layer_to_channels(conv3d(1,2,3), channels="all")
    model2.add_layer(flatten())
    model2.add_layer(dense(40, 2))

