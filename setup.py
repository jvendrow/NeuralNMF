from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()


setup(
    name='NeuralNMF',
    version='1.0.0',
    license='MIT',
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Joshua Vendrow",
    author_email='jvendrow@ucla.edu',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research ",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/jvendrow/NeuralNMF',
    keywords='neural nmf',
    install_requires=[
          'torch',
          'matplotlib',
          'scipy',
          'numpy',
          'fnnls',
          'tqdm',
      ],

)

