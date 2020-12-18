"""Tests for the ``plasticc`` module"""

import os
from pathlib import Path
from unittest import TestCase

import numpy as np
import sncosmo
from numpy.testing import assert_equal

from snat_sim import plasticc
from snat_sim.filters import register_lsst_filters
from snat_sim.lc_simulation import calc_x0_for_z
from tests.mock import create_mock_plasticc_light_curve

register_lsst_filters(force=True)
test_data_dir = Path(__file__).parent / 'plasticc_data'

# Used to store environmental variables on module setup / teardown
_OLD_ENV_VALUE = None
_ENVIRON_VAR_NAME = 'CADENCE_SIMS'


def setUpModule():
    """Set the ``CADENCE_SIMS`` variable in the environment"""

    global _OLD_ENV_VALUE
    _OLD_ENV_VALUE = os.environ.get(_ENVIRON_VAR_NAME, None)
    os.environ[_ENVIRON_VAR_NAME] = str(test_data_dir)


def tearDownModule():
    """Restore the ``CADENCE_SIMS`` variable to it's value before testing"""

    del os.environ[_ENVIRON_VAR_NAME]
    if _OLD_ENV_VALUE:
        os.environ[_ENVIRON_VAR_NAME] = _OLD_ENV_VALUE


class GetAvailableCadences(TestCase):
    """Tests for the ``get_available_cadences`` function"""

    def test_cadences_match_test_data(self):
        """Test returned cadences match those available in the test data"""

        test_data_cadences = [d.name for d in test_data_dir.glob('*') if d.is_dir()]
        self.assertEqual(plasticc.get_available_cadences(), test_data_cadences)


class GetModelHeaders(TestCase):
    """Tests for the ``get_model_headers`` function"""

    def test_correct_headers_for_test_data(self):
        """Test the returned list is empty for a cadence with no available data"""

        header_paths = plasticc.get_model_headers('alt_sched', model=11)
        file_names = sorted(path.name for path in header_paths)
        known_headers = ['LSST_WFD_NONIa-0004_HEAD.FITS', 'LSST_WFD_NONIa-0005_HEAD.FITS']
        self.assertListEqual(file_names, known_headers)


class CountLightCurves(TestCase):
    """Tests for the ``count_light_curves`` function"""

    test_cadence = 'alt_sched'
    test_model = 11
    lc_num_for_cadence = 8

    def test_lc_count_matches_test_data(self):
        """Test the number of counted light curves matches those in the test data"""

        counted_light_curves = plasticc.count_light_curves(self.test_cadence, self.test_model)
        self.assertEqual(counted_light_curves, self.lc_num_for_cadence)


class IterLCForHeader(TestCase):
    """Tests for the ``iter_lc_for_header`` function"""

    def test_lc_has_meta_data(self):
        """Test returned light curves have meta data"""

        test_header = plasticc.get_model_headers('alt_sched', 11)[0]
        lc = next(plasticc.iter_lc_for_header(test_header, verbose=False))
        self.assertTrue(lc.meta)


class IterLcForCadenceModel(TestCase):
    """Tests for the ``iter_lc_for_cadence_model`` function"""

    def test_lc_count_matches_count_light_curves_func(self):
        """Test returned light curve count matches the values returned by ``count_light_curves``"""

        total_lc_count = sum(1 for _ in plasticc.iter_lc_for_cadence_model('alt_sched', 11))
        expected_count = plasticc.count_light_curves('alt_sched', 11)
        self.assertEqual(total_lc_count, expected_count)


class FormatPlasticcSncosmo(TestCase):
    """Tests for the ``format_plasticc_sncosmo`` function"""

    def setUp(self):
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
