"""Copy a small subset of PLaSTIC data into the test suite.

Usage:
   python make_test_data.py [PLaSTICC Data Directory] [Cadence Name] [Model Number]

Example:
   python make_test_data.py /mnt/md0/sn-sims/ alt_sched 11
"""

from pathlib import Path
import sys
from astropy.io import fits

use_cadence = sys.argv[2]
use_model = int(sys.argv[3])

parent_sim_data_dir = Path(sys.argv[1]) / use_cadence / f'LSST_WFD_{use_cadence}_MODEL{use_model}'
test_data_dir = Path(__file__).parent / use_cadence / f'LSST_WFD_{use_cadence}_MODEL{use_model}'
test_data_dir.mkdir(parents=True)

# Keep four light-curves from each simulation file
for sim_file_num in (4, 5):
    # Create the header file
    hdul_head = fits.open(parent_sim_data_dir / f'LSST_WFD_NONIa-{sim_file_num:04}_HEAD.FITS')
    hdul_head[1].data = hdul_head[1].data[0: 4]
    hdul_head.writeto(test_data_dir / f'LSST_WFD_NONIa-{sim_file_num:04}_HEAD.FITS')

    # Create the photometry file
    hdul_phot = fits.open(parent_sim_data_dir / f'LSST_WFD_NONIa-{sim_file_num:04}_PHOT.FITS')
    data_end_index = hdul_head[1].data['PTROBS_MAX'][-1]
    hdul_phot[1].data = hdul_phot[1].data[:data_end_index]
    hdul_phot.writeto(test_data_dir / f'LSST_WFD_NONIa-{sim_file_num:04}_PHOT.FITS')
