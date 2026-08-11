"""
Setup script for MS-HGNN package
"""

from setuptools import setup, find_packages

setup(
    name='ms-hgnn',
    version='1.0.0',
    description='Multi-Scale Hierarchical Graph Neural Network for NSCLC Prognosis',
    author='Imam Dad, Jianfeng He',
    author_email='jfenghe@kust.edu.cn',
    packages=find_packages(),
    install_requires=[
        'torch>=1.12.0',
        'torch-geometric>=2.2.0',
        'numpy>=1.21.0',
        'scipy>=1.9.0',
        'pandas>=1.4.0',
        'scikit-learn>=1.1.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'pyyaml>=6.0',
        'h5py>=3.7.0',
        'tqdm>=4.64.0',
    ],
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
