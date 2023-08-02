# -*- coding: utf-8 -*-
# @Time    : 2023/4/24 9:15

from setuptools import setup, find_packages

install_requires = [
    'fastdatasets>=0.9.14 , <= 0.10',
    'numpy',
    'tqdm',
    'six'
],

setup(
    name='numpy-io',
    version='0.0.8',
    description='an easy training architecture',
    long_description='numpy-io: https://github.com/ssbuild/numpy_io.git',
    license='Apache License 2.0',
    url='https://github.com/ssbuild/numpy_io',
    author='ssbuild',
    author_email='9727464@qq.com',
    install_requires=install_requires,
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    package_data={"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx"]},
)
