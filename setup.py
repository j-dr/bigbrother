from distutils.core import setup

setup(
    name='bigbrother',
    version='0.1dev',
    packages=['bigbrother',],
    scripts=['bin/bb-validate', 'bin/bb-genhtml', 'bin/bb-compare'],
    long_description=open('README.md').read(),
    )
