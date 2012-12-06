slic-python
===========

SLIC Superpixel implementation wrapper for Python.

BUGFIX
------

There was a sort of major bug because I misunderstood the ARGB format used.
The last channel, not the first, should be ignored.
I simplified the functions now.
