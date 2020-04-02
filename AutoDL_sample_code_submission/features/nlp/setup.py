#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2020/2/29 20:43
# @Author:  Mecthew

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("tfidf",
                         ["tfidf.pyx"],
                         libraries=["m"],
                         extra_compile_args=["-ffast-math"])]
setup(
    name='tfidf',
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules)
