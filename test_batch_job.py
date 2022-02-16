import numpy as np
import astropy.units as u
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path

from roman_cgi_iefc import cgi
import misc

import proper
proper.prop_use_fftw(DISABLE=False)
proper.prop_fftw_wisdom( 1024 )

data_dir = Path('/groups/douglase/kians-data-files/roman-cgi-iefc-data')
flatmaps_dir = Path.home()/'src/pyfalco/roman/flatmaps'

dm1_flatmap = fits.getdata(flatmaps_dir/'dm1_m_flat_hlc_band1.fits')
dm2_flatmap = fits.getdata(flatmaps_dir/'dm2_m_flat_hlc_band1.fits')
dm1_design = fits.getdata(flatmaps_dir/'dm1_m_design_hlc_band1.fits')
dm2_design = fits.getdata(flatmaps_dir/'dm2_m_design_hlc_band1.fits')


hlci = cgi.CGI_PROPER(use_opds=True, use_fieldstop=True, quiet=True)

hlci.set_dm1(dm1_flatmap)
hlci.set_dm2(dm2_flatmap)
psf_flatmap = hlci.calc_psf()

hlci.set_dm1(dm1_design)
hlci.set_dm2(dm2_design)
psf_design = hlci.calc_psf()

misc.myimshow2(psf_flatmap, psf_design)
plt.savefig('testing_batch_script.png')

hlci.reset_dms()
dm1s = [dm1_flatmap, dm1_design]
dm2s = [dm2_flatmap, dm2_design]
psfs = hlci.calc_psfs(dm1s, dm2s)
misc.myimshow2(psfs[0], psfs[1], lognorm1=True, lognorm2=True)
plt.savefig('testing_batch_script_2.png')

print('Script done running.')

