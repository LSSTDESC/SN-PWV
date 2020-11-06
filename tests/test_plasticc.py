"""Tests for the ``plasticc`` module"""

from unittest import TestCase

import numpy as np
import sncosmo
from astropy.table import Table
from numpy.testing import assert_equal

from snat_sim import plasticc
from snat_sim.filters import register_lsst_filters
from snat_sim.lc_simulation import calc_x0_for_z

register_lsst_filters(force=True)


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
            'SIM_PEAKMJD': 0,
            'SIM_SALT2x1': .1,
            'SIM_SALT2c': .2,
            'SIM_REDSHIFT_CMB': .5,
            'SIM_SALT2x0': 1
        }
    )


class GetModelHeaders(TestCase):
    """Tests for the ``get_model_headers`` function"""

    def test_empty_list_for_no_local_data(self):
        """Test the returned list is empty for a cadence with no available data"""

        self.assertListEqual(
            [], plasticc.get_model_headers('fake_cadence', model=11),
            'Returned list is not empty')


class FormatPlasticcSncosmo(TestCase):
    """Tests for the ``format_plasticc_sncosmo`` function"""

    def setUp(self):
        """Create a mock PLaSTICC light-curve"""

        self.plasticc_lc = create_mock_plasticc_light_curve()
        self.formatted_lc = plasticc.format_plasticc_sncosmo(self.plasticc_lc)

    def test_correct_column_names(self):
        """Test the formatted data table has the correct columns"""

        expected_names = ['time', 'band', 'flux', 'fluxerr', 'zp', 'photflag', 'zpsys']
        self.assertSequenceEqual(self.formatted_lc.colnames, expected_names)

    def test_preserves_meta_data(self):
        """Test the formatted data table has the same metadata as the input table"""

        self.assertDictEqual(self.formatted_lc.meta, self.plasticc_lc.meta)


class ExtractCadenceData(TestCase):
    """Tests for the ``extract_cadence_data`` function"""

    def setUp(self):
        """Create a mock PLaSTICC light-curve"""

        self.plasticc_lc = create_mock_plasticc_light_curve()
        self.extracted_cadence = plasticc.extract_cadence_data(self.plasticc_lc)

    def test_correct_column_names(self):
        """Test the formatted data table has the correct columns"""

        expected_names = ['time', 'band', 'zp', 'zpsys', 'gain', 'skynoise']
        self.assertSequenceEqual(self.extracted_cadence.colnames, expected_names)

    def test_zp_is_overwritten(self):
        """Test the zero_point in the returned table is overwritten with a constant"""
        expected_zp = 25
        assert_equal(expected_zp, self.extracted_cadence['zp'])

    def test_filter_names_are_formatted(self):
        """Test filter names are formatted for use with sncosmo"""

        is_lower = all(f.islower() for f in self.extracted_cadence['band'])
        self.assertTrue(is_lower, 'Filter names include uppercase letters')

        is_prefixed = all(f.startswith('lsst_hardware_') for f in self.extracted_cadence['band'])
        self.assertTrue(is_prefixed, 'Filter names do not start with ``lsst_hardware_``')

    def test_drop_nondetection(self):
        """Test ``drop_nondetection=True`` removes non detections"""

        extracted_cadence = plasticc.extract_cadence_data(self.plasticc_lc, drop_nondetection=True)
        returned_dates = extracted_cadence['time']
        expected_dates = self.plasticc_lc[self.plasticc_lc['PHOTFLAG'] != 0]['MJD']
        assert_equal(returned_dates, expected_dates)


class DuplicatePlasticcSncosmo(TestCase):
    """Tests for the ``duplicate_plasticc_sncosmo`` function"""

    def setUp(self):
        self.model = sncosmo.Model('salt2-extended')
        self.plasticc_lc = create_mock_plasticc_light_curve()
        self.param_mapping = {  # Maps sncosmo param names to plasticc names
            't0': 'SIM_PEAKMJD',
            'x1': 'SIM_SALT2x1',
            'c': 'SIM_SALT2c',
            'z': 'SIM_REDSHIFT_CMB',
            'x0': 'SIM_SALT2x0'
        }

    def test_lc_meta_matches_params(self):
        """Test parameters in returned meta data match the input light_curve"""

        duplicated_lc = plasticc.duplicate_plasticc_sncosmo(self.plasticc_lc, model=self.model, cosmo=None)
        for sncosmo_param, plasticc_param in self.param_mapping.items():
            self.assertEqual(
                duplicated_lc.meta[sncosmo_param], self.plasticc_lc.meta[plasticc_param],
                f'Incorrect {sncosmo_param} ({plasticc_param}) parameter'
            )

    def test_x0_overwritten_by_cosmo_arg(self):
        """Test the x0 parameter is overwritten according to the given cosmology"""

        duplicated_lc = plasticc.duplicate_plasticc_sncosmo(self.plasticc_lc, model=self.model)
        expected_x0 = calc_x0_for_z(duplicated_lc.meta['z'], source=self.model.source)
        np.testing.assert_allclose(expected_x0, duplicated_lc.meta['x0'])
