# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for the ``plasticc`` module"""

from unittest import TestCase

import numpy as np
from astropy.table import Table
from numpy.testing import assert_equal

from sn_analysis import plasticc
from sn_analysis.plasticc import get_available_cadences

local_data_is_available = bool(get_available_cadences())


def create_mock_plasticc_light_curve():
    """Create a mock light-curve in the PLaSTICC data format

    Returns:
        An astropy table
    """

    time_values = np.arange(-20, 52)
    return Table({
        'MJD': time_values,
        'FLT': list('ugrizY') * (len(time_values) // 6),
        'FLUXCAL': np.ones_like(time_values),
        'FLUXCALERR': np.full_like(time_values, .2),
        'ZEROPT': np.full_like(time_values, 30),
        'PHOTFLAG': [0] * 10 + [6144] + [4096] * 61,
        'SKY_SIG': np.full_like(time_values, 80)
    })


class GetModelHeaders(TestCase):
    """Tests for the ``get_model_headers`` function"""

    def test_empty_list_for_no_local_data(self):
        """Test the returned list is empty for a cadence with no available data"""

        self.assertListEqual(
            [], plasticc.get_model_headers('fake_cadence'),
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
