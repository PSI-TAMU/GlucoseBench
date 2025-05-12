from setuptools import setup, find_packages

setup(
    name='glucosebench',
    version='0.1',
    packages=find_packages(include=['glucosebench', 'glucosebench.*']),
    install_requires=[
        'numpy', 'pandas', 'matplotlib', 'shapely'
    ],
    description='A toolkit for evaluating glucose prediction models',
)