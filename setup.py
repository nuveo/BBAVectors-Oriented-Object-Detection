#!/usr/bin/env python
from setuptools import setup
from bbavectors import __version__
from distutils.core import setup

long_description = open('README.md').read()

requirements = [
    'attrdict==2.0.1',
    'torch==1.6.0',
    'torchvision',
    'opencv-python==4.5.3.56',
    'Shapely==1.7.1',
    'matplotlib==3.0.3',
    'yacs==0.1.8',
    'tqdm>=4.46.0',
    'Jinja2==3.0.1'
]

setup(
    # Metadata
    name='bbavectors',
    version=__version__,
    author='Yi, Jingru and Wu, Pengxiang and Liu, Bo and Huang, Qiaoying and Qu, Hui and Metaxas, Dimitris',
    url='https://github.com/yijingru/BBAVectors-Oriented-Object-Detection',
    description='Oriented Object Detection in Aerial Images with Box Boundary-Aware Vectors',
    long_description=long_description,
    license='MIT',
    # Package info
    packages=['bbavectors', 'DOTA_devkit'],
    zip_safe=True,
    include_package_data=True,
    install_requires=requirements
)
