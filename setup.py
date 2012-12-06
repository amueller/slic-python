from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy.distutils.misc_util import get_numpy_include_dirs

setup(cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("slic", ["_slic.pyx", "SLIC.cpp"],
        language="c++",
        include_dirs=get_numpy_include_dirs())])
