from setuptools import setup, find_packages

setup(
    name='tdm',
    version='0.1',
    description='PyTorch Deep Models',
    author='Matvey Gerasyov',
    packages=find_packages(exclude=['tests']),
    install_requires=['torch >= 1.4', 'torchvision'],
    python_requires='>=3.6',
)
