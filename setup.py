from setuptools import setup, find_packages

setup(
    name='zero',
    version='0.0.1',
    description='Machine learning python package',
    url='https://github.com/dead4an/zero-nn',
    packages=find_packages(),
    requires=[
        'numpy',
        'pandas',
        'pyarrow'
    ]
)