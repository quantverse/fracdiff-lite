# Fracdiff: Super-fast Fractional Differentiation

This is lite version (no sklearn / torch) with relaxed dependencies.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Documentation](https://fracdiff.github.io/fracdiff/)

***Fracdiff*** performs fractional differentiation of time-series,
a la "Advances in Financial Machine Learning" by M. Prado.
Fractional differentiation processes time-series to a stationary one while preserving memory in the original time-series.
Fracdiff features super-fast computation and scikit-learn compatible API.

![spx](./examples/fig/spx.png)

## What is fractional differentiation?

See [M. L. Prado, "Advances in Financial Machine Learning"][prado].

## Features

### Functionalities

- [`fdiff`][doc-fdiff]: A function that extends [`numpy.diff`](https://numpy.org/doc/stable/reference/generated/numpy.diff.html) to fractional differentiation.

[doc-fdiff]: https://fracdiff.github.io/fracdiff/generated/fracdiff.fdiff.html

### Speed

Fracdiff is blazingly fast.

The following graphs show that *Fracdiff* computes fractional differentiation much faster than the "official" implementation.

It is especially noteworthy that execution time does not increase significantly as the number of time-steps (`n_samples`) increases, thanks to NumPy engine.

![time](https://user-images.githubusercontent.com/24503967/128821902-d38c2f46-989c-44e7-bd71-899f95553696.png)

The following tables of execution times (in unit of ms) show that *Fracdiff* can be ~10000 times faster than the "official" implementation.

|   n_samples | fracdiff        | official            |
|------------:|:----------------|:--------------------|
|         100 | 0.675 +-0.086   | 20.008 +-1.472      |
|        1000 | 5.081 +-0.426   | 135.613 +-3.415     |
|       10000 | 50.644 +-0.574  | 1310.033 +-17.708   |
|      100000 | 519.969 +-8.166 | 13113.457 +-105.274 |

|   n_features | fracdiff       | official             |
|-------------:|:---------------|:---------------------|
|            1 | 5.081 +-0.426  | 135.613 +-3.415      |
|           10 | 6.146 +-0.247  | 1350.161 +-15.195    |
|          100 | 6.903 +-0.654  | 13675.023 +-193.960  |
|         1000 | 13.783 +-0.700 | 136610.030 +-540.572 |

(Run on Ubuntu 20.04, Intel(R) Xeon(R) CPU E5-2673 v3 @ 2.40GHz. See [fracdiff/benchmark](https://github.com/fracdiff/benchmark/releases/tag/1115171075) for details.)

## How to use

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fracdiff/fracdiff/blob/main/examples/example_howto.ipynb)

### Fractional differentiation

A function [`fdiff`](https://fracdiff.github.io/fracdiff/#fdiff) calculates fractional differentiation.
This is an extension of `numpy.diff` to a fractional order.

```python
import numpy as np
from fracdiff import fdiff

a = np.array([1, 2, 4, 7, 0])
fdiff(a, 0.5)
# array([ 1.       ,  1.5      ,  2.875    ,  4.6875   , -4.1640625])
np.array_equal(fdiff(a, n=1), np.diff(a, n=1))
# True

a = np.array([[1, 3, 6, 10], [0, 5, 6, 8]])
fdiff(a, 0.5, axis=0)
# array([[ 1. ,  3. ,  6. , 10. ],
#        [-0.5,  3.5,  3. ,  3. ]])
fdiff(a, 0.5, axis=-1)
# array([[1.    , 2.5   , 4.375 , 6.5625],
#        [0.    , 5.    , 3.5   , 4.375 ]])
```

## Contributing

Any contributions are more than welcome.

The maintainer (simaki) is not making further enhancements and appreciates pull requests to make them.
See [Issue](https://github.com/fracdiff/fracdiff/issues) for proposed features.
Please take a look at [CONTRIBUTING.md](.github/CONTRIBUTING.md) before creating a pull request.

## References

- [Marcos Lopez de Prado, "Advances in Financial Machine Learning", Wiley, (2018).][prado]

[prado]: https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
