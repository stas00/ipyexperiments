from setuptools import setup, find_packages

# note: version is maintained inside ipyexperiments/version.py
exec(open('ipyexperiments/version.py').read())

with open("README.md", "r") as fh: long_description = fh.read()

def to_list(buffer): return list(filter(None, map(str.strip, buffer.splitlines())))

requirements = to_list("""
  ipython
  pynvml
  psutil
  pynvx>=1.0.0;platform_system=="Darwin"
  ipython>=6.0.0 # events support
""")

setup_requirements = ['pytest-runner']

test_requirements = to_list("""
  pytest
  pytest-ipynb
""")

setup(
    name = 'ipyexperiments',
    version = __version__,

    packages = find_packages(),
    include_package_data = True,

    install_requires = requirements,
    setup_requires   = setup_requirements,
    tests_require    = test_requirements,
    python_requires  = '>=3.6',

    test_suite = 'tests',

    license = "Apache License 2.0",

    description = "jupyter/ipython experiment containers for GPU and general RAM re-use and memory leaks detection.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = 'https://github.com/stas00/ipyexperiments',
    keywords = 'ipyexperiments, jupyter, ipython, memory, gpu',

    author = "Stas Bekman",
    author_email = 'stas@stason.org',

    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    zip_safe = False,
)
