from setuptools import setup

setup(name='autosum',
      version='0.1',
      description='Generate context-based embeddings of scientific literature',
      url='http://github.com/alingupta/autosum',
      author='Alind Gupta',
      license='MIT',
      packages=['autosum'],
      install_requires=[
          'requests',
          'bs4',
          'tensorflow',
          'matplotlib',
          'numpy',
          'nltk'
      ],
      include_package_data=True)
