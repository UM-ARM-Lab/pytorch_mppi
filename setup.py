from setuptools import setup

setup(
    name='pytorch_mppi',
    version='0.4.0',
    packages=['pytorch_mppi'],
    url='https://github.com/LemonPi/pytorch_mppi',
    license='MIT',
    author='zhsh',
    author_email='zhsh@umich.edu',
    description='Model Predictive Path Integral (MPPI) implemented in pytorch',
    install_requires=[
        'torch',
        'numpy'
    ],
    tests_require=[
        'gym<=0.20',
        'pygame'
    ]
)
