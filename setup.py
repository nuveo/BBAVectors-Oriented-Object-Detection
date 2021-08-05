#!/usr/bin/env python
from setuptools import setup, find_packages
from bbavectors import __version__
from distutils.core import setup

long_description = open('README.md').read()

requirements = [
    'attrdict==2.0.1',
    'opencv-python==4.5.3.56',
    'Shapely==1.7.1',
    'matplotlib>=3.1.3',
    'yacs==0.1.8',
    'tqdm>=4.46.0',
    'Jinja2==3.0.1'
]

extras = {
    'infer': [
        'torch==1.6.0',
        'torchvision',
    ],
}

setup(
    # Metadata
    name='bbavectors',
    entry_points=dict(
        console_scripts=["bbavectors=bbavectors.app.main:main"]),
    version=__version__,
    author='Yi, Jingru and Wu, Pengxiang and Liu, Bo and Huang, Qiaoying and Qu, Hui and Metaxas, Dimitris',
    url='https://github.com/yijingru/BBAVectors-Oriented-Object-Detection',
    description='Oriented Object Detection in Aerial Images with Box Boundary-Aware Vectors',
    long_description=long_description,
    license='MIT',
    # Package info
    packages=find_packages(),
    zip_safe=True,
    include_package_data=True,
    install_requires=requirements,
    extras_require=extras
)
