from setuptools import setup, find_packages, Extension

#######################################
# Prepare list of compiled extensions #
#######################################

extensions = []


#########
# Setup #
#########

setup(
    name='PyPLINE',
    version='0.0.1',
    description='Pipelined paralelisation based on MPI combining tracking capabilities of PyHEADTAIL and xsuite',
    url='https://github.com/PyCOMPLETE/PyPLINE',
    author='Xavier Buffat',
    packages=find_packages(), # finds all the packages in the folder
    ext_modules = extensions,
    install_requires=[
        'numpy>=1.0',
        'xobjects>=0.0.4',
        'xtrack>=0.0.1',
        'xfields>=0.0.1'
        ]
    )
