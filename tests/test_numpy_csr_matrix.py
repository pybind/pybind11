import pytest
from pybind11_tests import numpy_csr_matrix as m

pytestmark = pytest.requires_numpy

with pytest.suppress(ImportError):
    import numpy as np
    from scipy import sparse


def test_accept_csr_matrix():

    seed = 1398
    np.random.seed(seed)

    n_samples = 40
    n_features = 20
    sparsity = 1e-4

    features = sparse.rand(n_samples, n_features, density=sparsity, format='csr')
    m.accept_csr_matrix(features)

    assert all([1])
