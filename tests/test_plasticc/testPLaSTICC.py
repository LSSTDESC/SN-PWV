"""Tests for the ``snat_sim.plasticc.PLaSTICC`` class"""

from unittest import TestCase

from snat_sim.plasticc import PLaSTICC
from tests.mock import create_mock_light_curve


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
        returned_light_curves = len(list(self.dao.iter_cadence(verbose=False)))
        self.assertEqual(returned_light_curves, counted_light_curves)

class IterLcForCadenceModel(SetUp, TestCase):
    """Tests for the iteration of light-curves"""

    def test_lc_count_matches_count_light_curves_func(self) -> None:
        """Test returned light curve count matches the values returned by ``count_light_curves``"""

        total_lc_count = sum(1 for _ in self.dao.iter_cadence(verbose=False))
        expected_count = self.dao.count_light_curves()
        self.assertEqual(expected_count, total_lc_count)

    def test_iter_limit(self):
        """Test the number of returned light-curves is limited by the ``iter_lim`` argument"""

        total_lc_count = sum(1 for _ in self.dao.iter_cadence(iter_lim=5, verbose=False))
        self.assertEqual(5, total_lc_count)
