
from setuptools import setup, find_packages
import pathlib
import os

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
with open(os.path.join(here, 'README.md')) as f:
    long_description = f.read()

# Get the long description from the README file
with open(os.path.join(here, 'requirements.txt')) as f:
    requirements = f.read().split('\n')


# Set version
MAJOR = 0
MINOR = 0
MICRO = 0

VERSION = "{}.{}.{}".format(MAJOR,MINOR,MICRO)

# Dependencies
INSTALL_REQUIRES = requirements

SETUP_REQUIRES = [
]

TESTS_REQUIRE = [
    'pytest >= 3.6',
    'pytest-cov',
    'pytest-mock',
]

setup(
    name='demoapp',
    version=VERSION,
    description='Demo python application with container deployment',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hpaquet/ContainerDeployment',
    author='Hugo Paquet',
    author_email='paquet.hugo@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        #'License :: OSI Approved :: MIT License',
        #'Programming Language :: Python :: 3',
        #'Programming Language :: Python :: 3.5',
        #'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        #'Programming Language :: Python :: 3.8',
    ],
    keywords='sample, setuptools, development',
    packages=find_packages(),

    # Dependencies
    python_requires='>=3.7',
    install_requires=INSTALL_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    tests_require=TESTS_REQUIRE,
)