from setuptools import setup, find_packages

setup(
    name='tdm',
    version='0.1',
    description='Unofficial PyTorch implementation of some key deep models',
    author='Matvey Gerasyov',
    packages=find_packages(exclude=['tests']),
    install_requires=['torch >= 1.4', 'torchvision'],
    python_requires='>=3.6',
)
