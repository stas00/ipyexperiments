#!/usr/bin/env python

from setuptools import setup, find_packages

with open('README.md') as readme_file: readme = readme_file.read()

def to_list(buffer): return list(filter(None, map(str.strip, buffer.splitlines())))

requirements = to_list("""
  ipython
  nvidia-ml-py3
  psutil
""")
setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Stas Bekman",
    author_email='stas@stason.org',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="experiment containers for jupyter/ipython",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme,
    include_package_data=True,
    keywords='ipyexperiments',
    name='ipyexperiments',
    packages=find_packages(include=['ipyexperiments']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/stas00/ipyexperiments',
    version='0.1.0',
    zip_safe=False,
)
