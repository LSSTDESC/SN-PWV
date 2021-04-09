"""Functions for building objects used by the test suite"""

from datetime import datetime, timedelta
from typing import *

import numpy as np
import pandas as pd
from astropy.table import Table

from snat_sim.models import PWVModel
from snat_sim.models.supernova import ObservedCadence, SNModel
from snat_sim.pipeline.data_model import PipelinePacket


def create_mock_pwv_data(
        start_time=datetime(2020, 1, 1),
        end_time=datetime(2020, 12, 31),
        delta=timedelta(days=1),
        offset=timedelta(hours=0)):
    """Return a ``Series`` of mock PWV values that alternate between .5 and 1

    Args:
        start_time (datetime): Start time of the returned series index
        end_time   (datetime): End time of the returned series index
        delta     (timedelta): Sampling rate of the datetime values
        offset    (timedelta): Apply a linear offset to the returned values

    Returns:
        A pandas ``Series`` object
    """

    index = np.arange(start_time, end_time, delta).astype(datetime) + offset
    pwv = np.ones_like(index, dtype=float)
    pwv[::2] = 0.5
    return pd.Series(pwv, index=index)


def create_constant_pwv_model(constant_pwv_value=4):
    """Create a ``PWVModel`` instance that returns a constant PWV at zenith

    Args:
        constant_pwv_value (float): The PWV value to use as the constant

    Returns:
        A ``PWVModel`` instance
    """

    date_sampling = np.arange(datetime(2020, 1, 1), datetime(2020, 12, 31), timedelta(days=1))
    pwv = np.full(len(date_sampling), constant_pwv_value)
    model_data = pd.Series(pwv, index=pd.to_datetime(date_sampling))
    return PWVModel(model_data)


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
            'SNID': '123456',
            'RA': 10,
            'DECL': -5,
            'SIM_PEAKMJD': 0,
            'SIM_SALT2x1': .1,
            'SIM_SALT2c': .2,
            'SIM_REDSHIFT_CMB': .5,
            'SIM_SALT2x0': 1
        }
    )


def create_mock_cadence(
        obs_time: Collection[float] = range(-20, 51),
        bands: Collection[str] = ('decam_g', 'decam_r', 'decam_i', 'decam_z', 'decam_y'),
        zp: Union[int, float] = 25,
        zpsys: str = 'AB',
        gain: int = 1
) -> ObservedCadence:
    """Create a cadence object for a uniform observation cadence across multiple bands

    In the resulting cadence each of the given bands is observed at every time value.

    Args:
        obs_time: Array of phase values to include
        bands: Array of bands to include
        zp: The zero point
        zpsys: The zero point system
        gain: The simulated gain

    Returns:
        An Observed Cadence instance
    """

    all_times = np.concatenate([obs_time for _ in bands])
    band = np.concatenate([np.full_like(obs_time, b, dtype='U1000') for b in bands])
    return ObservedCadence(
        obs_times=all_times,
        bands=band,
        zp=zp,
        zpsys=zpsys,
        gain=gain,
        skynoise=0
    )


def create_mock_pipeline_packet(
        snid: int = 123456, include_lc: bool = True, include_fit: bool = True
) -> PipelinePacket:
    """Create a ``PipelinePacket`` instance with mock data

    Args:
        snid: The unique id value for the pipeline packet
        include_lc: Include a simulated light_curve in the packet
        include_fit: Include fit results for the simulated light_curve

    Returns:
        A ``PipelinePacket`` instance
    """

    sim_params, cadence = ObservedCadence.from_plasticc(create_mock_plasticc_light_curve())
    packet = PipelinePacket(snid, cadence=cadence, sim_params=sim_params)

    if include_lc:
        model = SNModel('salt2-extended')
        model.update({p: v for p, v in sim_params.items() if p in model.param_names})
        packet.light_curve = model.simulate_lc(cadence)

        if include_fit:
            packet.fit_result, packet.fitted_model = model.fit_lc(packet.light_curve, ['x0', 'x1', 'c'])
            packet.covariance = packet.fit_result.salt_covariance_linear()

    return packet
