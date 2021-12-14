import numpy as np
import xarray as xr

import dask
import dask.array as dsa


def empty_dask_array(shape, dtype=float, chunks=None):
    """A dask array that errors if you try to compute it
    Stolen from https://github.com/xgcm/xhistogram/blob/master/xhistogram/test/fixtures.py
    """

    def raise_if_computed():
        raise ValueError("Triggered forbidden computation on dask array")

    a = dsa.from_delayed(dask.delayed(raise_if_computed)(), shape, dtype)
    if chunks is not None:
        a = a.rechunk(chunks)
    return a


def example_da(shape, wrap="numpy"):
    """An example DataArray with the first dimension named "time"

        Parameters
        ----------
        shape : tuple
            The shape of the DataArray
        wrap : string
            | "numpy"; output DataArray wraps a numpy array
            | "dask"; output DataArray wraps a dask array
            | "dask_nocompute"; output DataArray wraps a dask array that errors if \
                computed
    """
    coords = [range(s) for s in shape]
    dims = ["time"] + [f"dim_{i}" for i in range(1, len(shape))]
    data = np.random.random(size=shape)
    if wrap == "dask_nocompute":
        data = empty_dask_array(shape)
    elif wrap == "dask":
        data = dsa.from_array(data)
    return xr.DataArray(data, coords=coords, dims=dims)
