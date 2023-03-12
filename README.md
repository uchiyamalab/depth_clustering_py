# Depth Clustering

This is a Python port of the Depth Clustering algorithm originally written in C++. The original GitHub repository can be found at:
- https://github.com/PRBonn/depth_clustering

Please note that the current implementation lacks some features of the original code, such as `DepthGroundRemover`. We will consider porting those if there are any requests.

If you are comfortable with C++, we recommend using the original implementation.

## Install

Prerequisites:

- Python >= 3.8
- numpy >= 1.23.1
- numba >= 0.56.2

You can install this `depth_clustering` library with the command below:

```
$ pip install git+https://github.com/uchiyamalab/depth_clustering_py.git@main
```

## How to use?

Please refer to the [/notebooks/examples.ipynb](/notebooks/examples.ipynb).

## Why we ported from the original C++ code to Python

The author worked at a new media art lab and learned about Depth Clustering while working on 3D LiDAR projects. Unfortunately, we needed to run the algorithm on multiple student computers with different environments (including M1 Mac, Windows, and Raspberry Pi), which required much effort to prepare the C++ build environments. As a solution, we ported the algorithm to Python. While Python code is generally much slower than C++ code, we found that using [Numba](https://numba.pydata.org/), a just-in-time (JIT) compiler based on LLVM, made the code relatively fast.


## Related publications

Please cite [the original author's papers](https://github.com/PRBonn/depth_clustering#related-publications) if you use this code.

## License

MIT License

Copyright (C) 2023  T. Kamatani

Copyright (C) 2020  I. Bogoslavskyi, C. Stachniss

Ported to Python by Takahiro Kamatani in [Uchiyama Lab, Institute of Art and Design, University of Tsukuba](https://www.geijutsu.tsukuba.ac.jp/uchiyamalab/).
