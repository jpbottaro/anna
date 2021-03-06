from __future__ import print_function
from setuptools import setup
from setuptools.command.test import test as TestCommand
import io
import os
import sys

import anna

here = os.path.abspath(os.path.dirname(__file__))


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.md')


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ["-c", "tests/pytest.ini"]
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

setup(
    name='anna',
    version=anna.__version__,
    url='http://github.com/jpbottaro/anna/',
    license='MIT',
    author='Juan Pablo Bottaro',
    tests_require=['pytest'],
    install_requires=['Tensorflow==2.3.1'],
    cmdclass={'test': PyTest},
    author_email='jpbottaro@gmail.com',
    description='NN experiments on traditional datasets.',
    long_description=long_description,
    packages=['anna'],
    include_package_data=True,
    platforms='any',
    test_suite='anna.test.test_anna',
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        ],
    extras_require={
        'testing': ['pytest'],
    }
)
