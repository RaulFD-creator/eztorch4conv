"""
Unit and regression test for the eztorch4conv package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

#import eztorch4conv
from eztorch4conv import *

def test_eztorch4conv_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "eztorch4conv" in sys.modules
    model = DCNN("model1", "./")
    model.add_layers([conv3d(neurons=2, input_shape=(3,9,9,9), conv_kernel=3),
                    flatten(),
                    dense(40)])
    model.add_callbacks([
                    early_stop(metric=["accuracy"],target=4),
                    checkpoint(metric=["accuracy"],target=4)])
    
    model2 = MCDCNN("model2", "./", 3)
    model2.add_layers_to_channels(layer=conv3d(neurons=2, input_shape=(3,9,9,9), conv_kernel=3),
                                 channels="all")
    model2.add_layers([flatten()])
 
    

