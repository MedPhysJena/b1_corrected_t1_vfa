# Function to fit T1 from a VFA acquisition with B1 correction

Minimal repository containing the python code to perform T1 fitting using VFA acquisition compensating for the known B1 inhomogeneity.

This repository is set up as an installable single-file python module. To install it, run from the root of the repository:

```sh
pip install -e .
```
This will install the dependencies (`numpy` and `scipy`) and will make the module available, letting you use in your code:

``` python
from b1_corrected_t1_vfa import fit_t1_vfa
```

Interested users may install the dependencies for the test and run the test

```sh
pip install -e ".[test]"
pytest
```

`matplotlib` is not considered to be a test dependency, but the test will produce visual outputs if `matplotlib` is available.
