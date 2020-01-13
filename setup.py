# -*- coding: utf-8 -*-
# -------------------------------
# Author: Zikang Xiong
# Email: zikangxiong@gmail.com
# Date:   2020-01-10 12:06:05
# Last Modified by:   Zikang Xiong
# Last Modified time: 2020-01-11 12:19:40
# -------------------------------
from setuptools import setup, find_packages

setup(
    name='VRL',
    version='1.0',
    description='PLDI\'19 Verifiable Reinforcement Learning',
    author='He Zhu, Zikang Xiong',
    author_email='he.zhu.cs@rutgers.edu, zikangxiong@gmail.com',
    maintainer='Zikang Xiong',
    maintainer_email='zikangxiong@gmail.com',
    url='https://github.com/caffett/VRL_CodeReview',
    packages=find_packages(),

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],

    install_requires=[
        'tensorflow>=1.13.0',
        'tflearn>=0.3.2', 
        'numpy',
        'scipy',
    ]
)

