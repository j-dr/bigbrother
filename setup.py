from distutils.core import setup

setup(
    name='bigbrother',
    version='0.1dev',
    packages=['bigbrother',],
    scripts=['bin/validate'],
    long_description=open('README.md').read(),
    )
