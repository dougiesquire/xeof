import pytest

import numpy as np
import numpy.testing as npt
import xarray as xr

import xeof
from xeof import eof, project_onto_eof
from .fixtures import example_da

from eofs.xarray import Eof


@pytest.mark.parametrize("shape", [(20, 10), (20, 5, 4)])
@pytest.mark.parametrize("n_modes", [20, 10])
@pytest.mark.parametrize("weight", ["none", "sqrt_cos_lat", "random"])
@pytest.mark.parametrize("wrap", ["numpy", "dask"])
def test_eof_values(shape, n_modes, weight, wrap):
    """Test values relative to Eof package"""
    data = example_da(shape, wrap=wrap)
    lat_dim = f"dim_{len(shape)-1}"
    xeof.core.LAT_NAME = lat_dim
    sensor_dims = [f"dim_{i}" for i in range(1, len(shape))]

    if weight == "none":
        weights = None
    elif weight == "sqrt_cos_lat":
        weights = np.cos(data[lat_dim] * np.pi / 180) ** 0.5
    elif weight == "random":
        weights = data.isel(time=0).copy()
        weight = weights.compute()

    res = eof(
        data,
        sensor_dims=sensor_dims,
        sample_dim="time",
        weight=weight,
        n_modes=n_modes,
        norm_PCs=False,
    )

    Eof_solver = Eof(data, weights=weights, center=False)
    ver_pcs = Eof_solver.pcs(pcscaling=0, npcs=n_modes)
    ver_eofs = Eof_solver.eofs(eofscaling=0, neofs=n_modes)
    ver_EV = Eof_solver.varianceFraction(neigs=n_modes)

    npt.assert_allclose(abs(res["pc"]), abs(ver_pcs))
    npt.assert_allclose(abs(res["eof"]), abs(ver_eofs))
    npt.assert_allclose(res["explained_var"], ver_EV)


@pytest.mark.parametrize("shape", [(20, 10), (20, 5, 4)])
@pytest.mark.parametrize("n_modes", [20, 10])
@pytest.mark.parametrize("wrap", ["numpy", "dask"])
def test_pc_normalisation(shape, n_modes, wrap):
    """Test normalisation of PCs"""
    data = example_da(shape, wrap=wrap)
    sensor_dims = [f"dim_{i}" for i in range(1, len(shape))]

    res = eof(
        data,
        sensor_dims=sensor_dims,
        sample_dim="time",
        weight="none",
        n_modes=n_modes,
        norm_PCs=True,
    )

    I_res = np.dot(res["pc"].values.T, res["pc"].values)
    I_ver = np.identity(res.sizes["mode"])

    npt.assert_allclose(I_res, I_ver, atol=1e-10)


@pytest.mark.parametrize("n_variables", [1, 2])
@pytest.mark.parametrize("wrap", ["numpy", "dask"])
def test_Dataset(n_variables, wrap):
    """Test Dataset input"""
    shape = (10, 5, 4)
    sensor_dims = [f"dim_{i}" for i in range(1, len(shape))]
    data = xr.Dataset(
        {f"var_{i}": example_da(shape, wrap=wrap) for i in range(n_variables)}
    )

    eof(data, sensor_dims=sensor_dims, sample_dim="time")


@pytest.mark.parametrize("shape", [(20, 10), (20, 5, 4)])
@pytest.mark.parametrize("weight", ["none", "sqrt_cos_lat"])
@pytest.mark.parametrize("wrap", ["numpy", "dask"])
def test_project_onto_eof_roundtrip(shape, weight, wrap):
    """Test projection onto fields used to calculate eofs"""

    data = example_da(shape, wrap=wrap)
    lat_dim = f"dim_{len(shape)-1}"
    xeof.core.LAT_NAME = lat_dim

    sensor_dims = [f"dim_{i}" for i in range(1, len(shape))]
    ver = eof(
        data,
        sensor_dims=sensor_dims,
        sample_dim="time",
        weight=weight,
        n_modes=np.inf,
        norm_PCs=False,
    )

    pc_res = project_onto_eof(data, ver["eof"], sensor_dims=sensor_dims, weight=weight)
    pc_ver = ver["pc"]

    npt.assert_allclose(pc_res - pc_ver, 0.0, atol=1e-10)


@pytest.mark.parametrize("shape", [(20, 10), (20, 5, 4)])
@pytest.mark.parametrize("weight", ["none", "sqrt_cos_lat"])
@pytest.mark.parametrize(
    "wrap", ["numpy"]
)  # Eof package sometimes thinks there are nans in dask arrays
def test_project_onto_eof_values(shape, weight, wrap):
    """Test projection onto fields used to calculate eofs"""

    data = example_da(shape, wrap=wrap)
    field = example_da(shape, wrap=wrap)
    lat_dim = f"dim_{len(shape)-1}"
    xeof.core.LAT_NAME = lat_dim

    if weight == "none":
        weights = None
    elif weight == "sqrt_cos_lat":
        weights = np.cos(data[lat_dim] * np.pi / 180) ** 0.5

    sensor_dims = [f"dim_{i}" for i in range(1, len(shape))]
    eofs = eof(
        data,
        sensor_dims=sensor_dims,
        sample_dim="time",
        weight=weight,
        n_modes=np.inf,
        norm_PCs=False,
    )

    pc_res = project_onto_eof(
        field, eofs["eof"], sensor_dims=sensor_dims, weight=weight
    )

    Eof_solver = Eof(data, weights=weights, center=False)
    pc_ver = Eof_solver.projectField(field)
    # Our mode start at 1:
    pc_ver = pc_ver.assign_coords({"mode": pc_ver["mode"] + 1})

    npt.assert_allclose(pc_res - pc_ver, 0.0, atol=1e-10)
