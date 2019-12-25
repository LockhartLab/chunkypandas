"""
test_dataframe.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""

import chunkypandas as cpd

from hypothesis import given, settings
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import unittest

# Hypothesis settings
settings.register_profile('lenient', deadline=None)
settings.load_profile('lenient')


# Hypotheses
chunksizes = st.sampled_from([None, 10, 100, 1000, 10000])


# Test chunkypandas.core.dataframe.py
class TestDataFrame(unittest.TestCase):
    @given(chunksizes, st.one_of(st.integers(), st.floats(), st.booleans()))
    def test_add_constant(self, chunksize, constant):
        # Get data
        df, cdf = _get_data(chunksize)

        # Add constant
        add1 = df.add(constant)
        add2 = cdf.add(constant)

        # Assert equal
        np.testing.assert_array_almost_equal(add1, add2)

    @given(chunksizes)
    def test_add_chunky(self, chunksize):
        # Get data
        df, cdf = _get_data(chunksize)

        # Add self
        add1 = df.add(df)
        add2 = cdf.add(cdf)

        # Assert equal
        np.testing.assert_array_almost_equal(add1, add2)

    @given(chunksizes)
    def test_count(self, chunksize):
        # Get data
        df, cdf = _get_data(chunksize)

        # Count!
        count1 = df.count()
        count2 = cdf.count()

        # Assert equal
        np.testing.assert_array_equal(count1, count2)


# Helper function to get data
def _get_data(chunksize):
    # Create DataFrame
    n_rows = 1000
    n_columns = 10
    df = pd.DataFrame(np.random.rand(n_rows, n_columns), columns=['x' + str(i) for i in range(n_columns)])

    # Create chunky DataFrame
    cdf = cpd.ChunkyDataFrame().from_pandas(df, chunksize=chunksize)

    # Return
    return df, cdf
