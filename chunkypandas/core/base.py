"""
base.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""

from chunkypandas.utilities import get_named_tempfile

from abc import ABC
import numpy as np
import os
import pandas as pd
from typelike import ListLike


# Base class for Chunky objects
class ChunkyBase(ABC):
    """
    Base class for Chunky objects. Not meant to be directly used.
    """

    # Initialize object
    def __init__(self, path, ext, chunksize, index_col):
        # TODO expand this with more pandas IO options; bare bones for MVP
        self.path = path
        self.ext = _get_ext(path, ext)
        self.chunksize = chunksize
        self.index_col = index_col

    # Add
    def __add__(self, other):
        return self.add(other)

    # What happens when the class instance is called
    def __repr__(self):
        return self.to_pandas().__repr__()

    # Get chunks
    def _get_chunks(self):
        """
        Get chunks by reading file through pandas
        """

        # If ext is CSV
        if self.ext == 'csv':
            chunks = pd.read_csv(self.path, chunksize=self.chunksize, index_col=self.index_col, header=0)

        # Otherwise, we don't know how to read the file
        else:
            raise AttributeError('cannot read extension {}'.format(self.ext))

        # Return
        return chunks if self.chunksize is not None else [chunks]

    # Add self to other
    def add(self, other):
        """
        Add to the ChunkySeries

        Currently supported are constant values and other ChunkySeries instances

        Parameters
        ----------
        other : NumberLike or ChunkySeries
            Something to add to this instance of ChunkySeries

        Returns
        -------
        ChunkySeries
            This instance added to *other*
        """

        # Return
        return self.compute_combine_reduce(compute='add', combine='concat', other=other)

    # Compute, combine, reduce
    # TODO should other arguments be allowed?
    def compute_combine_reduce(self, compute='skip', combine='skip', reduce='skip', other=None):
        """
        Compute, combine, and reduce chunks

        Pre-defined functions include:
        * **add** : adds two things together
        * **concat** : concatenates LikeLike objects
        * **empty** : returns None for step
        * **skip** : returns input for step
        * **sub** : subtracts first thing by second thing

        Parameters
        ----------
        compute : callable or str
            Function applied to pandas DataFrame or Series chunks, or string used to represent pre-defined functions.
            (Default: 'skip', i.e., no operation is performed on pandas objects).
        combine : callable or str
            Function used to combine chunks, or string used to represent pre-defined functions. (Default: 'skip', i.e.,
            chunks are not combined and the last chunk will be returned).
        reduce : callable or str
            Function used to provide final transformation, or string used to represented pre-defined functions.
            (Default: 'skip', i.e., the result of all the combines is returned).
        other : NumberLike or child of ChunkyBase
            Some compute functions perform mathematical operations involved an `other` thing

        Returns
        -------
        result
        """

        # Define compute, combine, reduce
        compute = _define_compute_combine_reduce(compute)
        combine = _define_compute_combine_reduce(combine)
        reduce = _define_compute_combine_reduce(reduce)

        # Get chunks from self
        chunks = self._get_chunks()

        # Specify empty result set
        result = None

        # If other is not None and is a Chunky object
        # TODO don't want to over-engineer this, but perhaps this could be simplified since the for-loop bulk is same
        if isinstance(other, ChunkyBase):
            for chunk1, chunk2 in zip(chunks, other._get_chunks()):
                if not np.array_equal(chunk1.index.values, chunk2.index.values):
                    raise AttributeError('indices between chunks should match')
                _result = compute(chunk1, chunk2)
                if result is None:
                    result = _result
                else:
                    result = combine(result, _result)

        # Otherwise, if other is not None
        elif other is not None:
            for chunk in chunks:
                _result = compute(chunk, other)
                if result is None:
                    result = _result
                else:
                    result = combine(result, _result)

        # Finally, if other is None
        else:
            for chunk in chunks:
                _result = compute(chunk)
                if result is None:
                    result = _result
                else:
                    result = combine(result, _result)

        # Return reduced version
        return reduce(result)

    # Compute count
    def count(self):
        return self.compute_combine_reduce(compute='count', combine='add')

    # Head -- for now let's hack this; in this future we can make this more intelligent by computing n_chunks
    def head(self, n=10):
        # How many chunks do we need?
        # n_chunks = np.ceil(n / self.chunksize)

        # Save old chunksize and set chunksize to n
        old_chunksize = self.chunksize
        self.chunksize = n

        # Without any arguments, we simply get the first chunk
        result = self.compute_combine_reduce()

        # Set chunksize back to old_chunksize
        self.chunksize = old_chunksize

        # Return result
        return result

    # Compute mean
    def mean(self):
        """
        Compute mean
        """

        # Compute
        def _compute(chunk):
            return chunk.sum(), chunk.count()

        # Reduce
        def _reduce(result):
            return result[0] / result[1]

        # Go!
        return self.compute_combine_reduce(compute=_compute, combine='add', reduce=_reduce)

    # From pandas
    def from_pandas(self, pandas_object, chunksize=1000):
        """
        Convert `pandas` to `chunkypandas`

        Parameters
        ----------
        pandas_object : pandas.DataFrame or pandas.Series
            Pandas object to convert
        chunksize : int
            Size of chunks to process for new Chunky object

        Returns
        -------
        ChunkyDataFrame or ChunkySeries
            Pandas object converted to Chunky object
        """

        # Get temporary file
        path = get_named_tempfile()

        # Write pandas object to temporary file
        pandas_object.to_csv(path, index=True, header=True)

        # Reinitialize self
        # TODO is this bad form?
        self.__init__(path, ext='csv', chunksize=chunksize, index_col=0)

        # Finally, return self
        return self

    # Convert to pandas
    def to_pandas(self):
        """
        Convert `chunkypandas` to `pandas`

        Returns
        -------
        pandas.DataFrame or pandas.Series
            Pandas DataFrame or Series depending if ChunkyDataFrame or ChunkySeries
        """

        # Go!
        return self.compute_combine_reduce(combine='concat')


# Add
def _add(x, y):
    # If x is from pandas, we can make use of its method
    if isinstance(x, (pd.DataFrame, pd.Series)):
        result = x.add(y)

    # If both x and y are ListLike
    elif isinstance(x, ListLike) and isinstance(y, ListLike):
        print(x, y)
        # TODO remove eventually
        assert len(x) == len(y)
        result = []
        for i in range(len(x)):
            result.append(x[i] + y[i])

    # Otherwise, perform simply
    else:
        result = x + y

    # Return
    return result


# Count
def _count(x, y=None):
    # If x is pandas, we can make use of its method
    if isinstance(x, (pd.DataFrame, pd.Series)):
        result = x.count()

    # Otherwise, get length?
    else:
        result = len(x)

    # Return
    return result


def _get_ext(path, ext):
    # If ext is None, parse
    if path is not None and ext is None:
        ext = os.path.splitext(path)[1][1:]

    # If ext is string, convert to lowercase
    if isinstance(ext, str):
        ext = ext.lower()

    # Return
    return ext


# Pandas concat
# TODO we need to choose to ignore_index intelligently
def _concat(x, y):
    return pd.concat([x, y], ignore_index=True, axis=0)


# Empty
def _empty(x, y=None):
    return None


# Define compute, combine, reduce
def _define_compute_combine_reduce(thing):
    # List of functions
    functions = {
        'add': _add,
        'concat': _concat,
        'count': _count,
        'empty': _empty,
        'skip': _skip,
        'sub': _sub,
    }

    # If thing is a callable, use it
    if callable(thing):
        result = thing

    # Else if thing is a string get function from list of functions
    elif isinstance(thing, str):
        result = functions[thing]

    # Finally, if we made it here, we don't know anything
    else:
        raise AttributeError('unexpected type {}'.format(type(thing)))

    # Return
    return result


# Skip
def _skip(x, y=None):
    return x


# Subtract
def _sub(x, y):
    # If y is from pandas, we can make use of its method
    if isinstance(x, (pd.DataFrame, pd.Series)):
        result = x.sub(y)

    # Otherwise, perform simply
    else:
        result = x - y

    # Return
    return result


# Empty
def _empty_combine(x, y):
    return None


# Empty reduce
def _empty_reduce(result):
    return None


# Skip compute
def _skip_compute(chunk):
    return chunk


# Skip combine
def _skip_combine(result, _result):
    return _result


# Skip reduce
def _skip_reduce(result):
    return result


# Subtract
def _sub(x, y):
    return x - y
