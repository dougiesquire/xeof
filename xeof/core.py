import numpy as np
import xarray as xr
import dask.array as darray


SENSOR_DIM_NAME = "sensor_dims"
MODE_DIM_NAME = "mode"
LAT_NAME = "lat"


def _apply_weights(ds, weight):
    """Apply weights to ds

    Parameters
    ----------
    ds : xarray DataArray or Dataset
        Array to use to compute EOFs.
    weight : str or xarray DataArray of Dataset
        Weighting to apply prior to svd. Two in-built options are
        currently provided that can be specified by providing strings:
        | "none"; no weighting applied
        | "sqrt_cos_lat"; cos(lat)^0.5 weighting
        Otherwise, an array of weights should be supplied
    """

    degtorad = np.pi / 180

    if isinstance(weight, str):
        if weight == "none":
            weight = 1
        elif weight == "sqrt_cos_lat":
            if LAT_NAME not in ds.dims:
                raise ValueError(
                    f"{LAT_NAME} is not a dimension of da. Please provide the name of the latitude "
                    + "dimension through xeof.core.LAT_NAME=<latitude dimension>"
                )
            else:
                weight = np.cos(ds[LAT_NAME] * degtorad) ** 0.5

    return weight * ds


def eof(
    ds, sensor_dims, sample_dim="time", weight="sqrt_cos_lat", n_modes=20, norm_PCs=True
):
    """ Returns the empirical orthogonal functions (EOFs), and associated principle component
        timeseries (PCs), and explained variances of provided array. Follows notation used in
        Bjornsson H. and Venegas S. A. 1997 A Manual for EOF and SVD analyses of Climatic Data,
        whereby, (phi, sqrt_lambdas, EOFs) = svd(data) and PCs = phi * sqrt_lambdas

        Parameters
        ----------
        ds : xarray DataArray or Dataset
            Array to use to compute EOFs.
        sensor_dims : str, optional
            EOFs sensor dimension. Usually 'lat' and 'lon'.
        sample_dim : str, optional
            EOFs sample dimension. Usually 'time'.
        weight : str or xarray DataArray of Dataset
            Weighting to apply prior to svd. Two in-built options are
            currently provided that can be specified by providing strings:
            | "none"; no weighting applied
            | "sqrt_cos_lat"; cos(lat)^0.5 weighting (default)
            Otherwise, an array of weights should be supplied
        n_modes : values, optional
            Number of EOF modes to return. If the number of modes requested
            is more than the number that are available, then all available
            modes will be returned.
        norm_PCs : boolean, optional
            If True, return the PCs normalised by sqrt(lambda) (ie phi), else return
            PCs = phi * sqrt(lambda)

        Returns
        -------
        eof : xarray Dataset
            | Dataset containing the following variables:
            | EOFs; array containing the empirical orthogonal functions
            | PCs; array containing the associated principle component timeseries
            | lambdas; array containing the eigenvalues of the covariance of the input data
            | explained_var; array containing the fraction of the total variance explained \
              by each EOF mode

        Examples
        --------
        >>> A = xr.DataArray(np.random.normal(size=(6,4,40)),
        ...                  coords=[('lat', np.arange(-75,76,30)), ('lon', np.arange(45,316,90)),
        ...                          ('time', pd.date_range('2000-01-01', periods=40, freq='M'))])
        >>> xeof.eof(A, sensor_dims=['lat','lon'])
        <xarray.Dataset>
        Dimensions:        (lat: 6, lon: 4, mode: 20, time: 40)
        Coordinates:
          * time           (time) datetime64[ns] 2000-01-31 2000-02-29 ... 2003-04-30
          * mode           (mode) int64 1 2 3 4 5 6 7 8 9 ... 12 13 14 15 16 17 18 19 20
          * lat            (lat) int64 -75 -45 -15 15 45 75
          * lon            (lon) int64 45 135 225 315
        Data variables:
            EOFs           (mode, lat, lon) float64 -0.05723 -0.01997 ... 0.08166
            PCs            (time, mode) float64 1.183 -1.107 -0.5385 ... -0.08552 0.1951
            lambdas        (mode) float64 87.76 80.37 68.5 58.14 ... 8.269 6.279 4.74
            explained_var  (mode) float64 0.1348 0.1234 0.1052 ... 0.009644 0.00728
    """

    def _svd(X, n_modes, norm_PCs):
        if isinstance(X, darray.core.Array):

            @darray.as_gufunc(
                signature="(sample,sensor)->(sample,mode),(mode),(mode,sensor)",
                output_dtypes=(float, float, float),
                output_sizes={
                    "sample": X.shape[-2],
                    "sensor": X.shape[-1],
                    "mode": min(X.shape[-2], X.shape[-1]),
                },
                allow_rechunk=True,
            )
            def _gu_svd(x):
                return np.linalg.svd(x, full_matrices=False)

            u, s, v = _gu_svd(X)
        else:
            u, s, v = np.linalg.svd(X, full_matrices=False)

        if norm_PCs:
            pcs = u
        else:
            pcs = np.swapaxes((np.swapaxes(u, -1, -2).T * s.T).T, -1, -2)
        eofs = v
        lambdas = s ** 2
        explained_var = (lambdas.T / np.sum(lambdas, axis=-1).T).T

        return (
            pcs[..., :n_modes],
            eofs[..., :n_modes, :],
            lambdas[..., :n_modes],
            explained_var[..., :n_modes],
        )

    if (not isinstance(ds, xr.DataArray)) & (not isinstance(ds, xr.Dataset)):
        raise ValueError("ds must be an xarray DataArray or Dataset")

    ds_weighted = _apply_weights(ds, weight)

    # Stack sample dimensions -----
    ds_weighted_stacked = ds_weighted.stack(**{SENSOR_DIM_NAME: sensor_dims})

    if n_modes > ds_weighted_stacked.sizes[SENSOR_DIM_NAME]:
        n_modes = ds_weighted_stacked.sizes[SENSOR_DIM_NAME]

    if n_modes > ds_weighted_stacked.sizes[sample_dim]:
        n_modes = ds_weighted_stacked.sizes[sample_dim]

    pc_dims = [sample_dim, MODE_DIM_NAME]
    eof_dims = [MODE_DIM_NAME, SENSOR_DIM_NAME]
    lambda_dims = [MODE_DIM_NAME]
    explained_var_dims = [MODE_DIM_NAME]
    output_core_dims = [pc_dims, eof_dims, lambda_dims, explained_var_dims]
    output_sizes = {MODE_DIM_NAME: n_modes}

    pcs, eofs, lambda_, explained_var = xr.apply_ufunc(
        _svd,
        *(ds_weighted_stacked, n_modes, norm_PCs),
        input_core_dims=[
            [
                sample_dim,
                SENSOR_DIM_NAME,
            ],
            [],
            [],
        ],
        output_core_dims=output_core_dims,
        output_sizes=output_sizes,
        output_dtypes=[float],
        dask="allowed",
    )
    if isinstance(ds, xr.DataArray):
        pcs = pcs.rename("pc")
        eofs = eofs.rename("eof")
        lambda_ = lambda_.rename("lambda")
        explained_var = explained_var.rename("explained_var")
    else:
        pcs = pcs.rename({var: f"pc__{var}" for var in pcs.data_vars})
        eofs = eofs.rename({var: f"eof__{var}" for var in eofs.data_vars})
        lambda_ = lambda_.rename({var: f"lambda__{var}" for var in lambda_.data_vars})
        explained_var = explained_var.rename(
            {var: f"explained_var__{var}" for var in explained_var.data_vars}
        )

    EOF = xr.merge([pcs, eofs, lambda_, explained_var])
    EOF[MODE_DIM_NAME] = np.arange(1, n_modes + 1)

    return EOF.unstack(SENSOR_DIM_NAME)


def project_onto_eof(field, eof, sensor_dims, weight=None):
    """Project a field onto a set of provided EOFs to generate a corresponding set of
    pseudo-PCs

    Parameters
    ----------
    field : xarray DataArray
        Array containing the data to project onto the EOFs
    eof : xarray DataArray
        Array contain set of EOFs to project onto.
    sensor_dims : str, optional
        EOFs sensor dimension.
    weight : xarray DataArray, optional
        Weighting to apply to field prior to projection. If weight=None, cos(lat)^2
        weighting are used.

    Returns
    -------
    projection : xarray DataArray
        Array containing the pseudo-PCs

    Examples
    --------
    >>> A = xr.DataArray(np.random.normal(size=(6,4,40)),
    ...                  coords=[('lat', np.arange(-75,76,30)), ('lon', np.arange(45,316,90)),
    ...                          ('time', pd.date_range('2000-01-01', periods=40, freq='M'))])
    >>> eof = xeof.eof(A, sensor_dims=['lat','lon'])
    >>> project_onto_eof(A, eof['eof'], sensor_dims=['lat','lon'])
    <xarray.DataArray (mode: 20, time: 40)>
    array([[ ... ]])
    Coordinates:
      * mode     (mode) int64 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
      * time     (time) datetime64[ns] 2000-01-31 2000-02-29 ... 2003-04-30

    """

    degtorad = np.pi / 180

    # Apply weights -----
    if weight is None:
        if LAT_NAME not in field.dims:
            raise ValueError(
                f"{LAT_NAME} is not a dimension of field. Please provide the "
                + "name of the latitude dimension through xeof.core.LAT_NAME "
                + "=<latitude dimension>"
            )
        else:
            weight = np.cos(field[LAT_NAME] * degtorad) ** 0.5
    field_weighted = weight.fillna(0) * field

    return xr.dot(eof, field_weighted, dims=sensor_dims)
