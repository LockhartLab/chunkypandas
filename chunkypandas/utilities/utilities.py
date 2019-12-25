"""
utilities.py
written in Python3
author: C. Lockhart
"""

import atexit
from glob import iglob
import os
from tempfile import gettempdir


# Clean up temporary flies
def clean():
    for file in iglob(os.path.join(gettempdir(), 'chunkypandas_*')):
        os.remove(file)


# At exit, clean!
atexit.register(clean)


# Get a named temporary file
def get_named_tempfile():
    return os.path.join(gettempdir(), 'chunkypandas_' + os.urandom(24).hex() + '.csv')
