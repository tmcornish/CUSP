#########################################################################
# Module containing convenience functions for calculating angular power
# spectra.
#########################################################################

import pymaster as nmt
import numpy as np


def get_bin_pairings(nbins, auto_only=False, labels=None):
    '''
    Returns pairs of IDs for each tomographic bin being analysed. Also
    returns comma-separated string versions of the ID pairs, or if specific
    labels have been provided will return those labels paired in the same
    order as the bin indices.

    Parameters
    ----------
    nbins: int
        Number of bins being considered.

    auto_only: bool
        If True, will only return each bin paired with itself.

    labels: list[str] or None
        (Optional) Specific labels to assign to each bin. If None, will
        return string versions of the indices assigned to each bin.

    Returns
    -------
    pairings: list[tuple[int]]
        List of possible bin pairings.

    pairings_s: list[str]
        List of comma-separated string versions of the ID pairs.

    label_pairs: list[tuple[str]]
        List of bin label pairings.
    '''
    import itertools

    lb = list(range(nbins))
    if auto_only:
        pairings = [(i, i) for i in lb]
    else:
        pairings = [i for i in itertools.product(lb, lb)
                    if tuple(reversed(i)) >= i]
    pairings_s = [f'{p[0]},{p[1]}' for p in pairings]

    if labels is None:
        return pairings, pairings_s
    else:
        label_pairs = [(labels[p[0]], labels[p[1]]) for p in pairings]
        return pairings, pairings_s, label_pairs


def get_bpw_edges(ell_max, ell_min=1, nbpws=10, spacing='linear'):
    '''
    Returns an array of bandpower edges for use in pseudo-Cl computation,
    based on the specified maximum and (optional) minimum multipole, and
    the type of spacing between bandpowers.

    Parameters
    ----------

    ell_max: int
        Maximum multipole to be considered.

    ell_min: int (optional)
        Minimum multipole to be considered (assumed to be 1 by default).

    nbpws: int
        Desired number of bandpowers.

    spacing: str (optional)
        String specifying the desired type of spacing between bandpowers. Must
        be one of either 'linear', 'log' or 'N19' (Nicola+19 bandpower edges).
        NOTE: selecting 'N19' will overwrite ell_min and nbpws.

    Returns
    -------
    bpw_edges: np.ndarray[int]
        Array containing the edges of each bandpower.
    '''
    if spacing == 'N19':
        bpw_edges = np.array([100, 200, 300, 400, 600, 800, 1000, 1400, 1800,
                              2200, 3000, 3800, 4600, 6200, 7800, 9400, 12600,
                              15800]).astype(int)
        # Remove any bin edges higher then ell_max
        bpw_edges = bpw_edges[bpw_edges <= ell_max]
    elif spacing == 'linear':
        bpw_edges = np.unique(
            np.linspace(ell_min, ell_max, nbpws+1).astype(int)
            )
    elif spacing == 'log':
        bpw_edges = np.unique(
            np.geomspace(ell_min, ell_max, nbpws+1).astype(int)
            )
    else:
        raise ValueError('spacing must be one of "linear", "log", or "N19".')

    # Check if any bins were lost due to spacing being too small
    n_removed = (nbpws + 1) - len(bpw_edges)
    if n_removed > 0:
        print(f'WARNING: {n_removed} bins were lost due to spacing between'
              'bins being too small.')

    return bpw_edges


def get_data_from_sacc(s, auto_only=False):
    '''
    Retrieves the relevant covariance information, which depends on the
    user settings in config.py.

    Parameters
    ----------
    s: sacc.sacc.Sacc
        Sacc object containing the angular power spectrum information.

    auto_only: bool
        If True, retrieves information for the autocorrelations only.

    Returns
    -------
    ells: np.ndarray
        Array of multipoles at which the power spectra are defined (i.e.
        effective multipoles of each bandpower).

    cells: np.ndarray
        Array of arrays containing the desired power spectra.

    cov: np.ndarray
        Matrix of the relevant convariances.
    '''

    # Get the tracer combinations
    combos = s.get_tracer_combinations()
    # Get the number of tomographic bins
    nbins = len(np.unique(combos))
    # Get the full covariance matrix from the Sacc
    cov_full = s.covariance.covmat

    if auto_only:
        # Cycle through the redshift bins
        ells, cells, inds = [], [], []
        for i in range(nbins):
            ells_now, cells_now, inds_now = s.get_ell_cl('cl_00',
                                                         f'bin_{i}',
                                                         f'bin_{i}',
                                                         return_ind=True)
            ells.append(ells_now)
            cells.append(cells_now)
            inds.extend(list(inds_now))
        # Use the returned indices to retrieve the relevant covariance info
        cov = cov_full[inds][:, inds]

    else:
        # Get data for all power spectra
        ell_cl = [s.get_ell_cl('cl_00', i, j) for i, j in combos]
        cov = cov_full
        # List the ells and cells for each pairing
        ells = [ell_cl[i][0] for i in range(len(ell_cl))]
        cells = [ell_cl[i][1] for i in range(len(ell_cl))]

    return ells, cells, cov


def select_from_sacc(s, tracer_combos, data_type):
    '''
    Given a Sacc object and a set of tracer combinations, will return a new
    Sacc object containing only the information for those tracer combinations.

    Parameters
    ----------
    s: sacc.sacc.Sacc or str
        The Sacc object containing the information for many tracers. If a
        string, must be the path to a Sacc file.

    tracer_combos: list[tuple]
        List of tuples, with each tuple containing a pair of tracer names.

    data_type: str
        Data type for which the information is to be extracted. E.g. 'cl_00' or
        'galaxy_density_cl'. Use print(sacc.standard_types) to see list of
        possible values.

    Returns
    -------
    s_new: sacc.sacc.Sacc
        Sacc object containing only the desired information.
    '''
    import sacc

    # Check if input is a string
    if type(s) is str:
        s = sacc.Sacc.load_fits(s)

    # Get the unique tracer names
    tc_unique = np.unique(tracer_combos)
    # Set up a new Sacc object and add tracers
    s_new = sacc.Sacc()
    for tc in tc_unique:
        s_new.add_tracer_object(s.get_tracer(tc))
    # Now add ell and C_ell info for each desired combination
    inds_all = []
    for tc in tracer_combos:
        ells, cells, inds = s.get_ell_cl(data_type, *tc, return_ind=True)
        # Get the window functions
        wins = s.get_bandpower_windows(inds)
        # Add the ell_cl info
        s_new.add_ell_cl(data_type, *tc, ells, cells, window=wins)
        # Add the indices to the list
        inds_all.extend(list(inds))
    # Add the covariance info
    if s.covariance is not None:
        s_new.covariance = sacc.covariance.FullCovariance(
            s.covariance.covmat[inds_all][:, inds_all]
            )
    else:
        s_new.covariance = None

    return s_new


def compute_covariance(w, cw, f_i1, f_i2, f_j1=None, f_j2=None, f_sky=None,
                       return_cl_coupled=False, return_cl_guess=False):
    '''
    Computes the Gaussian covariance for a pair of angular power spectra,
    given the fields used to compute said spectra.
    NOTE: this currently assumes the same mask is used for all fields
    involved.

    Parameters
    ----------
    w: pymaster.NmtWorkspace
        NaMaster workspace defined with the mask applied to all invovled
        fields.

    cw: pymaster.NmtCovarianceWorkspace
        Covariance workspace defined with the mask applied to all involved
        fields.

    f_i1, f_i2: pymaster.NmtField, pymaster.NmtField
        Fields contributing to the first power spectrum.

    f_j1, f_j2: pymaster.NmtField, pymaster.NmtField
        Fields contributing to the second power spectrum. If None, will be set
        to f_i1 and f_i2.

    f_sky: float
        Estimate of the observed sky fraction (assumed the same for all
        fields). If None, will calculate using the mask.

    return_cl_coupled: bool
        Whether to return the coupled C_ells computed as part of the covariance
        calculation.

    return_cl_guess: bool
        Whether to return the best-guess coupled C_ells computed as part of the
        covariance calculation.

    Returns
    -------
    covar: numpy.ndarray
        Covariance matrix for the pair of angular power spectra.

    err_cell: numpy.ndarray
        1-sigma uncertainty for the cross-correlation in each bandpower.

    cl_coupled_X1Y2: list[numpy.ndarray]
        (Optional) Coupled C_ells for each field pair.

    cl_guess_X1Y2: list[numpy.ndarray]
        (Optional) Best-guess coupled C_ells for each field pair.
    '''
    # See if additional fields have been provided for the second power spectrum
    if f_j1 is None:
        f_j1 = f_i1
    if f_j2 is None:
        f_j2 = f_i2

    # If no sky fraction estimate is provided, compute from mask
    if f_sky is None:
        f_sky = np.mean(f_i1.get_mask() ** 2.)

    # Compute coupled c_ells for each possible combination of i and j
    cl_coupled_i1j1 = nmt.compute_coupled_cell(f_i1, f_j1)
    cl_coupled_i1j2 = nmt.compute_coupled_cell(f_i1, f_j2)
    cl_coupled_i2j1 = nmt.compute_coupled_cell(f_i2, f_j1)
    cl_coupled_i2j2 = nmt.compute_coupled_cell(f_i2, f_j2)
    # Use these along with the mask to get a guess of the true C_ell
    cl_guess_i1j1 = cl_coupled_i1j1 / f_sky
    cl_guess_i1j2 = cl_coupled_i1j2 / f_sky
    cl_guess_i2j1 = cl_coupled_i2j1 / f_sky
    cl_guess_i2j2 = cl_coupled_i2j2 / f_sky

    covar = nmt.gaussian_covariance(cw,
                                    0, 0, 0, 0,			# Spin of each field
                                    [cl_guess_i1j1[0]],
                                    [cl_guess_i1j2[0]],
                                    [cl_guess_i2j1[0]],
                                    [cl_guess_i2j2[0]],
                                    w)
    # Errorbars for each bandpower
    err_cell = np.diag(covar) ** 0.5

    to_return = [covar, err_cell]
    if return_cl_coupled:
        to_return.append([cl_coupled_i1j1, cl_coupled_i1j2,
                          cl_coupled_i2j1, cl_coupled_i2j2])
    if return_cl_guess:
        to_return.append([cl_guess_i1j1, cl_guess_i1j2,
                          cl_guess_i2j1, cl_guess_i2j2])

    return (*to_return,)


def apply_scale_cuts(ells, cells, cov, lmin, lmax):
    '''
    Applies scale cuts to the data involved in the fit.

    Parameters
    ----------
    ells: list[numpy.ndarray]
        List of arrays containing the effective multipoles for each
        bin pairing.

    cells: list[numpy.ndarray]
        List of arrays containing the angular power spectra for each
        bin pairing.

    cov: numpy.ndarray or None
        Covariance matrix for the angular power spectra.

    lmin: int or list[int]
        Minimum multipole to include for each power spectrum. If a single
        value, will apply to all power spectra provided; otherwise, lmin
        must be a list with length equal to the number of power spectra.

    lmax: int or list[int]
        Maximum multipole to include for each power spectrum. If a single
        value, will apply to all power spectra provided; otherwise, lmin
        must be a list with length equal to the number of power spectra.

    labels: list[str] or None
        List of labels to assign to each scale cut. If None, will simply label
        each cut with an index ranging from 0 to N(C_ells)-1.

    Returns
    -------
    ells_cut: list[numpy.ndarray]
        List of effective multipoles after applying the scale cuts.

    cells_cut: list[numpy.ndarray]
        List of C_ells after applying the scale cuts.

    cov_cut: numpy.ndarray
        Covariance matrix after applying the scale cuts.
    '''

    # Determine number of power spectra provided
    if np.ndim(ells) > 1:
        N_cells = len(ells)
    else:
        N_cells = 1
        ells = [ells]
        cells = [cells]

    # If lmin and lmax are single values, fill arrays with them
    if type(lmin) is int:
        lmin = np.full(N_cells, lmin)
    if type(lmax) is int:
        lmax = np.full(N_cells, lmax)

    # Copy the inputs to avoid overwriting them
    ells_cut = ells.copy()
    cells_cut = cells.copy()
    cov_cut = cov.copy()

    # Set up a list for containing the masks for each bin pairing
    masks = []
    # Cycle through the bin pairings
    for i in range(N_cells):
        # Mask the multipoles above/below lmax/lmin
        lmask = (ells_cut[i] <= lmax[i]) * (ells_cut[i] > lmin[i])
        ells_cut[i] = ells_cut[i][lmask]
        cells_cut[i] = cells_cut[i][lmask]
        # Append the mask to the list
        masks.append(lmask)
    # To mask the covariance matrix, combine and flatten all masks, then
    # take the outer product with itself
    masks = np.array(masks).flatten()
    nkeep = int(masks.sum())
    covmask = np.outer(masks, masks)
    cov_cut = cov[covmask].reshape((nkeep, nkeep))

    return ells_cut, cells_cut, cov_cut
