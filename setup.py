# -*- coding: utf-8 -*-
# @Time    : 2023/4/24 9:15

from setuptools import setup, find_packages

ignore = ['test','tests']

setup(
    name='numpy-io',
    version='0.0.1',
    description='an easy training architecture',
    long_description='numpy-io: https://github.com/ssbuild/numpy-io.git',
    license='Apache License 2.0',
    url='https://github.com/ssbuild/numpy-io',
    author='ssbuild',
    author_email='9727464@qq.com',
    install_requires=[
        'fastdatasets>=0.9.6 , <= 1',
        'numpy',
        'tqdm',
        'six'
    ],
    packages=[p for p in find_packages() if p not in ignore]
)
