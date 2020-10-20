import sncosmo
from astropy.table import Table

from snat_sim import filters, plasticc, sn_magnitudes, modeling

filters.register_lsst_filters(force=True)


def iter_custom_lcs(
        model, cadence, iter_lim=None, gain=20, skynr=100, quality_callback=None, verbose=True):
    """Simulate light-curves for a given PLaSTICC cadence

    Args:
        model               (Model): Model to use in the simulations
        cadence               (str): Cadence to use when simulating light-curves
        gain                  (int): Gain to use during simulation
        skynr                 (int): Simulate skynoise by scaling plasticc ``SKY_SIG`` by 1 / skynr
        quality_callback (callable): Skip light-curves if this function returns False
        verbose              (bool): Display a progress bar
    """

    # model = copy(model)

    # Determine redshift limit of the given model
    u_band_low = sncosmo.get_bandpass('lsst_hardware_u').minwave()
    source_low = model.source.minwave()
    zlim = (u_band_low / source_low) - 1

    counter = -1
    iter_lim = float('inf') if iter_lim is None else iter_lim
    for light_curve in plasticc.iter_lc_for_cadence_model(cadence, model=11, verbose=verbose):
        counter += 1
        if counter >= iter_lim:
            break

        if light_curve.meta['SIM_REDSHIFT_CMB'] >= zlim:
            continue

        model.set(ra=light_curve.meta['RA'], dec=light_curve.meta['DECL'])
        duplicated_lc = plasticc.duplicate_plasticc_sncosmo(light_curve, model, gain=gain, skynr=skynr)

        if quality_callback and not quality_callback(duplicated_lc):
            continue

        yield duplicated_lc


def passes_quality_cuts(light_curve):
    """Return whether light-curve has 2+ two bands each with 1+ data point with SNR > 5

    Args:
        light_curve (Table): Astropy table with sncosmo formatted light-curve data

    Returns:
        A boolean
    """

    if light_curve.meta['z'] > .88:
        return False

    light_curve = light_curve.group_by('band')

    passed_cuts = []
    for band_lc in light_curve.groups:
        passed_cuts.append((band_lc['flux'] / band_lc['fluxerr'] > 5).any())

    return sum(passed_cuts) >= 2


pwv_interpolator = lambda *args: 5
variable_pwv_effect = modeling.VariablePWVTrans(pwv_interpolator)
variable_pwv_effect.set(res=5)

sn_model_with_pwv = modeling.Model(
    source='salt2-extended',
    effects=[variable_pwv_effect],
    effect_names=[''],
    effect_frames=['obs']
)

light_curves = iter_custom_lcs(
    sn_model_with_pwv, cadence='alt_sched', iter_lim=100, quality_callback=passes_quality_cuts)

if __name__ == '__main__':
    model_without_pwv = sncosmo.Model('salt2-extended')
    fitted_mag, fitted_params = sn_magnitudes.fit_mag(
        model=model_without_pwv,
        light_curves=light_curves,
        vparams=['x0', 'x1', 'c'],
        bands=['lsst_hardware_' + b for b in 'ugrizy'])
