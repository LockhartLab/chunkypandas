"""
series.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""

from .base import ChunkyBase

import numpy as np
import pandas as pd
from typelike import ListLike, NumberLike


# noinspection PyMissingConstructor
class ChunkySeries(ChunkyBase):
    """
    Tired of memory-intensive pandas objects? Look no further! With ``ChunkySeries``, you pretend like you're still
    using pandas but utilize your hard-drive instead of memory
    """

    # Initialize class instance
    def __init__(self, path, ext=None, chunksize=1000, index_col=0):
        """
        Initialize class instance

        Parameters
        ----------
        path : str
            Location to data
        ext : str
            (Optional) Data extension. If not provided, guessed
        chunksize : int
            Size of data chunks to read in (DEfault: 1000)
        index_col : int
            Location of index. None means no index (Default: 0)
        """

        super().__init__(path, ext, chunksize, index_col)

    # Add this ChunkySeries to some `other` thing
    def __add__(self, other):
        return self.add(other)

    # Subtract this ChunkySeries by some `other` thing
    def __sub__(self, other):
        return self.sub(other)

    # Subtract other from self
    def sub(self, other):
        """
        Subtract from the ChunkySeries

        Currently supported are constant values and other ChunkySeries instances

        Parameters
        ----------
        other : NumberLike or ChunkySeries
            Something to add to this instance of ChunkySeries

        Returns
        -------
        ChunkySeries
            This instance subtracted by *other*
        """

        # Return
        return self.compute_combine_reduce(compute='sub', combine='concat', other=other)
