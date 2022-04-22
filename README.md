Easy creation of 3D Deep Convolutional Neural Networks (3D-DCNN) or Multi-channel DCNN (3D-MCDCNN)
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/RaulFD-creator/eztorch4conv/workflows/CI/badge.svg)](https://github.com/RaulFD-creator/eztorch4conv/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/RaulFD-creator/eztorch4conv/branch/main/graph/badge.svg?token=U6t3PP1uZX)](https://codecov.io/gh/RaulFD-creator/eztorch4conv)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/RaulFD-creator/eztorch4conv.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/RaulFD-creator/eztorch4conv/context:python)


A Python package with out of the box pytorch classes to easily design and train DCNN and MC-DCNN models. 

Working on my Master's Thesis I found that there were not many easy to use implementations of MC-DCNN models or even DCNN in Pytorch. This package I hope will provide a clear user-friendly environment to simplify the Pytorch sintaxis by wrapping most of its functionality in easier to understand main classes. Phase 1 of development has been finished and the project can be installed by following this instructions:

1. Clone this repository in a directory:

```bash
git clone https://github.com/RaulFD-creator/eztorch4conv
```

2. Install the package using pip:

> pip install -e eztorch4conv/

3. Install the required dependencies either using pip:

> pip install pytorch pandas cudatoolkit

3b. Or using anaconda:

> conda install pytorch pandas cudatoolkit -c conda-forge


### Copyright

Copyright (c) 2022, Raúl Fernández Díaz


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
