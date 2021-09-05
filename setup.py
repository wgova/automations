from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

setup(
    name='ClusterValidation',
    version='0.1.1',
    description='Validate clustering results',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Webster Gova',
    author_email='webgster@yahoo.com',
    license='LICENSE.txt',
    install_requires=[
        'scikit-learn',
        'pandas',
        'numpy>=1.16.5',
        'seaborn',
        'matplotlib',
        'packaging',
        'pyclustering',
        'sklearn_extra'
    ],
    tests_require='pytest',
    setup_requires='pytest-runner',
    python_requires='>=3.5'
)
