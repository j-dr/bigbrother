# bigbrother

A package to run distributed validation metrics over large mock galaxy catalogs with an API designed to allow for integration with diversely formatted files.

## Dependencies

* numpy
* scipy
* astropy
* matplotlib
* fitsio
* corrfunc
* healpy
* cosmocalc
* helpers
* yaml

## Installation
For now the only way to install this code is:
``git clone https://github.com/j-dr/bigbrother.git``   
``cd bigbrother``   
``python setup.py install (--user)``


## Some Features

### API
bigbrother is designed to be agnostic to file formatting choices. In order to do this, the user is required to provide formatting information. The main object which handles this is the Ministry. A Ministry holds information about cosmology, redshift and area coverage of the catalog in question. It is also the object which coordinates the reading of data and running calculations on it.

Currently Ministry objects can handle galaxy and halo catalogs through galaxycatalog and halocatalog objects respectively. There are some built in types of galaxy and halo catalogs, but user defined catalogs are the most flexible and easiest to use for those not familiar with the specific build in catalog types. A user can define a catalog by providing the following information:

* filestruct : A dictionary whose keys define file types and values contain all the files of that type.
* fieldmap   : A dictionary whose keys are mapkeys and whose values are also  dictionaries. These sub-dictionaries have keys that are the column names of the files and values that are the file types containing those fields.
* unitmap    : A dictionary whose keys are mapkeys and whose values are the units of those fields.

Finally, the user must provide the validation metrics which they wish to run on the catalogs. Metrics are defined as objects which can be found in the submodules with metric in their names:

* magnitudemetric : galaxy magnitudes
* massmetric      : halo mass and other related halo quantities
* corrmetric      : correlation functions and other spatial statistics
* lineofsight     : redshift quantities
* densitymetric   : environmental density quantities

The metrics should then be provided by assigning a list of them to the metrics attribute of Ministry:

`` msr.metrics = [LuminosityFunction, MassFunction] ``

 These metrics will then be run on the provided files by calling the ministry function validate.

### Command line interface
It is also possible to run validations from the command line by defining the requisite quantities in a YAML config file. Look at validate.yaml to see an example set up. All of the information that can be provided via the normal API can be provided by specifying it in the config file.

In order to run the calculations defined in the config file, use the command:

`` bb-validate validate.yaml ``

This will call validate and store the ministry object as a pickle file. When run this way the user can enable parallelism by setting parallel equal to True in the Validate section of the config file. The parallelism is enabled via MPI, so when it is enabled ``bb-validate`` must be called ``mpirun`` or the equivalent call.
