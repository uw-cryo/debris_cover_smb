#!/usr/bin/env python

from distutils.core import setup

setup(name='debris_cover_smb',
      version='0.1',
      description='library for computing and analysing glacier surface mass balance from repeat satellite stereo DEMs',
      author='Shashank Bhushan and Team',
      author_email='sbaglapl@uw.edu',
      license='MIT',
      long_description=open('README.md').read(),
      url='https://github.com/uw-cryo/debris_cover_smb',
      packages=['debris_cover_smb'],
      install_requires=['requests']
      )