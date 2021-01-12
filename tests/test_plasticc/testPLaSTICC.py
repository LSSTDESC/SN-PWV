"""Tests for the ``plasticc`` module"""

from unittest import TestCase

from snat_sim.plasticc import PLaSTICC
from tests.mock import create_mock_plasticc_light_curve


class SetUp:
    """Generic setup tasks"""

    @classmethod
    def setUpClass(cls) -> None:
        """Instantiate a PLaSTICC data access object"""

        cls.test_cadence = 'alt_sched'
        cls.test_model = 11
        cls.dao = PLaSTICC(cls.test_cadence, cls.test_model)


class GetAvailableCadences(TestCase):
    """Test returned cadences match those packaged with ``snat_sim`` by default"""

    def runTest(self) -> None:
        self.assertEqual(PLaSTICC.get_available_cadences(), ['alt_sched'])


class GetModelHeaders(SetUp, TestCase):
    """Tests for the collection of header files"""

    def runTest(self) -> None:
        header_paths = self.dao.get_model_headers()
        file_names = sorted(path.name for path in header_paths)
        known_headers = ['LSST_WFD_NONIa-0004_HEAD.FITS', 'LSST_WFD_NONIa-0005_HEAD.FITS']
        self.assertListEqual(file_names, known_headers)


class CountLightCurves(SetUp, TestCase):
    """Test the number of counted light curves matches those in the test data"""

    def runTest(self) -> None:
        counted_light_curves = self.dao.count_light_curves()
        returned_light_curves = len(list(self.dao.iter_lc(verbose=False)))
        self.assertEqual(returned_light_curves, counted_light_curves)


class IterLCForHeader(SetUp, TestCase):
    """Test returned light curves have meta data"""

    def runTest(self) -> None:
        test_header = self.dao.get_model_headers()[0]
        lc = next(self.dao._iter_lc_for_header(test_header, verbose=False))
        self.assertTrue(lc.meta)


class IterLcForCadenceModel(SetUp, TestCase):
    """Tests for the iteration of light-curves"""

    def test_lc_count_matches_count_light_curves_func(self) -> None:
        """Test returned light curve count matches the values returned by ``count_light_curves``"""

        total_lc_count = sum(1 for _ in self.dao.iter_lc(verbose=False))
        expected_count = self.dao.count_light_curves()
        self.assertEqual(expected_count, total_lc_count)

    def test_iter_limit(self):
        """Test the number of returned light-curves is limited by the ``iter_lim`` argument"""

        total_lc_count = sum(1 for _ in self.dao.iter_lc(iter_lim=5, verbose=False))
        self.assertEqual(5, total_lc_count)


class FormatPlasticcSncosmo(SetUp, TestCase):
    """Tests for the formatting of PLaSTICC data into the sncosmo data model"""

    def setUp(self) -> None:
        self.plasticc_lc = create_mock_plasticc_light_curve()
        self.formatted_lc = self.dao.format_data_to_sncosmo(self.plasticc_lc)

    def test_correct_column_names(self) -> None:
        """Test the formatted data table has the correct columns"""

        expected_names = ['time', 'band', 'flux', 'fluxerr', 'zp', 'photflag', 'zpsys']
        self.assertSequenceEqual(self.formatted_lc.colnames, expected_names)

    def test_preserves_meta_data(self) -> None:
        """Test the formatted data table has the same metadata as the input table"""

        self.assertDictEqual(self.formatted_lc.meta, self.plasticc_lc.meta)
