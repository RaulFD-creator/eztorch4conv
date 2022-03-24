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
    model.add_many_layers(conv3d(2,3),
                    flatten(),
                    dense(40),
                    early_stop(metric=["accuracy"],target=4, model=model),
                    checkpoint(metric=["accuracy"],target=4), model=model)
    
    model2.add_layer_to_channels(layer=conv3d(2,3), channels="all")
    model2.add_layer(flatten())
    model2.add_layer(dense(40))
    

