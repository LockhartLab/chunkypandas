"""
dataframe.py
written in Python3
author: C. Lockhart <chris@lockhartlab.org>
"""

from .base import ChunkyBase
from .series import ChunkySeries

import os
import tempfile


# Create class
class ChunkyDataFrame(ChunkyBase):
    """
    `DataFrame` in `chunkypandas` offloads use of internal memory by increasing IO

    In other words, a typical `pandas` DataFrame is read as chunks, computed, and recompiled
    """

    # Initialize
    def __init__(self, path=None, ext=None, chunksize=1000, index_col=0):
        """
        Initialize class instance

        Parameters
        ----------
        path : str
            Location of file
        ext : str
            (Optional) File type of `path`
        chunksize : int
            The size of chunks to read in and handle in memory (Default: 1000)
        index_col : int
            Location of index in `path`. `None` means no index. (Default: 0)
        """

        # IO details
        super().__init__(path, ext, chunksize, index_col)

    # Get item returns ChunkySeries
    def __getitem__(self, column):
        return self._column_to_series(str(column))

    # Convert column to chunkypandas Series
    # TODO csv is probably sluggish ... see if you can use different file format for performance boost
    def _column_to_series(self, column):
        """
        Convert *column* to `chunkypandas.Series`

        Parameters
        ----------
        column : str
            Column to convert to Series

        Returns
        -------
        chunkypandas.Series
        """

        # Create path for temporary file
        path = os.path.join(tempfile.gettempdir(), 'chunkypandas_' + os.urandom(24).hex() + '.csv')

        # Save the column to the tempfile
        # TODO check that columns are the same?
        def _compute(chunk):
            header = False if os.path.exists(path) else True
            chunk[[column]].iloc[:, 0].to_csv(path, index=True, header=header, mode='a+')
            return None

        # Create a file that just includes column information
        self.compute_combine_reduce(compute=_compute, combine='empty', reduce='empty')

        # Return ChunkySeries
        return ChunkySeries(path, ext='csv', chunksize=self.chunksize, index_col=0)

    # TODO write this function
    def pivot_table(self, aggfunc='mean', *args, **kwargs):
        """
        Pivot table is tricky because chunks might have different values. We need to utilize the first chunks bins for
        all other chunks (?)

        Parameters
        ----------
        aggfunc
        args
        kwargs

        Returns
        -------

        """

        def _compute(chunk):
            if aggfunc == 'mean':
                result = (chunk.pivot_table(aggfunc='sum', *args, **kwargs),
                          chunk.pivot_table(aggfunc='count', *args, **kwargs))
            else:
                result = chunk.pivot_table(aggfunc=aggfunc, *args, **kwargs)
            return result

        def _reduce(result):
            if aggfunc == 'mean':
                result = result[0] / result[1]
            return result

        return self.compute_combine_reduce(compute=_compute, combine='add', reduce=_reduce)