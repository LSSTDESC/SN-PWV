"""Mock objects used when evaluating the test suite"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from astropy.table import Table

from snat_sim.models import PWVModel


def create_mock_pwv_data(
        start_time=datetime(2020, 1, 1),
        end_time=datetime(2020, 12, 31),
        delta=timedelta(days=1),
        offset=timedelta(hours=0)):
    """Return a ``Series`` of mock PWV values that alternate between .5 and 1

    Args:
        start_time (datetime): Start date of the returned series index
        end_time   (datetime): End date of the returned series index
        delta     (timedelta): Sampling rate of the datetime values
        offset    (timedelta): Apply a linear offset to the returned values

    Returns:
        A pandas ``Series`` object
    """

    index = np.arange(start_time, end_time, delta).astype(datetime) + offset
    pwv = np.ones_like(index, dtype=float)
    pwv[::2] = 0.5
    return pd.Series(pwv, index=index)


def create_constant_pwv_model(constant_pwv_value=4, cache_pwv_los=False):
    """Create a ``PWVModel`` instance that returns a constant PWV at zenith

    Args:
        constant_pwv_value (float): The PWV value to use as the constant
        cache_pwv_los        (int): Set ``cache_pwv_los`` for the returned ``PWVModel`` object

    Returns:
        A ``PWVModel`` instance
    """

    date_sampling = np.arange(datetime(2020, 1, 1), datetime(2020, 12, 31), timedelta(days=1))
    pwv = np.full(len(date_sampling), constant_pwv_value)
    model_data = pd.Series(pwv, index=pd.to_datetime(date_sampling))
    return PWVModel(model_data, cache_pwv_los=cache_pwv_los)


def create_mock_plasticc_light_curve():
    """Create a mock light-curve in the PLaSTICC data format

    Returns:
        An astropy table
    """

    time_values = np.arange(-20, 52)
    return Table(
        data={
            'MJD': time_values,
            'FLT': list('ugrizY') * (len(time_values) // 6),
            'FLUXCAL': np.ones_like(time_values),
            'FLUXCALERR': np.full_like(time_values, .2),
            'ZEROPT': np.full_like(time_values, 30),
            'PHOTFLAG': [0] * 10 + [6144] + [4096] * 61,
            'SKY_SIG': np.full_like(time_values, 80)
        },
        meta={
            'SNID': 123456,
            'RA': 10,
            'DECL': -5,
            'SIM_PEAKMJD': 0,
            'SIM_SALT2x1': .1,
            'SIM_SALT2c': .2,
            'SIM_REDSHIFT_CMB': .5,
            'SIM_SALT2x0': 1
        }
    )


def create_mock_pipeline_outputs():
    """Create DataFrame with mock results from the snat_sim pipeline using DES SN3YR data

    Returns:
        An astropy table if ``path`` is not given
    """

    from sndata.des import SN3YR

    # Download DES data if not already available
    sn3yr = SN3YR()
    sn3yr.download_module_data()

    # Load DES fit results and format them to match pipeline outputs
    pipeline_fits = sn3yr.load_table('SALT2mu_DES+LOWZ_C11.FITRES')
    pipeline_fits.rename_column('CID', 'snid')
    pipeline_fits.rename_column('zCMB', 'z')
    pipeline_fits.rename_column('zCMBERR', 'z_err')
    pipeline_fits.rename_column('PKMJD', 't0')
    pipeline_fits.rename_column('PKMJDERR', 't0_err')
    pipeline_fits.rename_column('x0ERR', 'x0_err')
    pipeline_fits.rename_column('x1ERR', 'x1_err')
    pipeline_fits.rename_column('cERR', 'c_err')
    pipeline_fits.rename_column('NDOF', 'dof')
    pipeline_fits.rename_column('FITCHI2', 'chisq')
    pipeline_fits.rename_column('RA', 'ra')
    pipeline_fits.rename_column('DECL', 'dec')
    pipeline_fits.rename_column('mB', 'mb')
    pipeline_fits.rename_column('mBERR', 'mb_err')
    pipeline_fits.meta = dict()

    # Keep only the data outputted by the snat_sim fitting pipeline
    keep_columns = ['snid', 'dof', 'chisq', 'ra', 'dec', 'mb', 'mb_err']
    for param in ['z', 't0', 'x0', 'x1', 'c']:
        keep_columns.append(param)
        keep_columns.append(param + '_err')
    pipeline_fits = pipeline_fits[keep_columns]
    return pipeline_fits.to_pandas()
