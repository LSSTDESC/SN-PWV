"""Tests for the ``snat_sim.models.light_curve.LightCurve`` class"""

from unittest import TestCase

import numpy as np
import pandas as pd
import sncosmo

from snat_sim.models.light_curve import LightCurve


class ParsingFromSncosmo(TestCase):
    """Test instance attributes match table data"""

    @staticmethod
    def runTest() -> None:
        data = sncosmo.load_example_data()
        light_curve = LightCurve.from_sncosmo(data)

        np.testing.assert_array_equal(data['time'], light_curve.time)
        np.testing.assert_array_equal(data['band'], light_curve.band)
        np.testing.assert_array_equal(data['flux'], light_curve.flux)
        np.testing.assert_array_equal(data['fluxerr'], light_curve.fluxerr)
        np.testing.assert_array_equal(data['zp'], light_curve.zp)
        np.testing.assert_array_equal(data['zpsys'], light_curve.zpsys)
        np.testing.assert_array_equal(0, light_curve.phot_flag)


class Casting(TestCase):
    """Tests for the casting of ``LightCurve`` instances to other data types"""

    @staticmethod
    def test_to_astropy() -> None:
        """Test data returned by ``to_astropy`` matches data used to build the LightCurve"""

        data = sncosmo.load_example_data()
        data['phot_flag'] = 0

        light_curve = LightCurve(
            time=data['time'],
            band=data['band'],
            flux=data['flux'],
            fluxerr=data['fluxerr'],
            zp=data['zp'],
            zpsys=data['zpsys'],
            phot_flag=data['phot_flag']
        )

        np.testing.assert_array_equal(data, light_curve.to_astropy())

    @staticmethod
    def test_to_pandas() -> None:
        """Test data returned by ``to_pandas`` matches data used to build the LightCurve"""

        data = sncosmo.load_example_data().to_pandas('time')
        data['phot_flag'] = 0.0

        light_curve = LightCurve(
            time=data.index,
            band=data['band'],
            flux=data['flux'],
            fluxerr=data['fluxerr'],
            zp=data['zp'],
            zpsys=data['zpsys'],
            phot_flag=data['phot_flag']
        )

        pd.testing.assert_frame_equal(data, light_curve.to_pandas())
