##############################################################################
# Module containing functions and variables for formatting plots.
##############################################################################


def check_for_latex():
    '''
    Checks if LaTeX is installed on machine; returns True if so and False
    otherwise.

    Returns
    -------
    installed: bool
        True if LaTeX is installed and False otherwise.
    '''
    import shutil

    if shutil.which('latex'):
        installed = True
    else:
        installed = False

    return installed


# Dictionary containing custom formatting for plots
styledict = {
    'figure.figsize': (6., 4.5),
    'legend.fontsize': 14,
    'legend.shadow': False,
    'legend.framealpha': 0.9,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'axes.labelsize': 24,
    'axes.linewidth': 2.,
    'image.origin': 'lower',
    'xtick.minor.visible': True,
    'xtick.major.size': 7,
    'xtick.minor.size': 4,
    'xtick.major.width': 2.,
    'xtick.minor.width': 1.5,
    'xtick.top': True,
    'xtick.major.top': True,
    'xtick.minor.top': True,
    'xtick.direction': 'in',
    'ytick.minor.visible': True,
    'ytick.major.size': 7,
    'ytick.minor.size': 4,
    'ytick.major.width': 2.,
    'ytick.minor.width': 1.5,
    'ytick.right': True,
    'ytick.major.right': True,
    'ytick.minor.right': True,
    'font.size': 22,
    'font.weight': 'bold',
    'ytick.direction': 'in',
    'text.usetex': check_for_latex(),  # Enables LaTeX fonts and symbols
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
    'axes.formatter.useoffset': False,
}

# Colours
red = '#eb0505'
dark_red = '#ab0000'
ruby = '#C90058'
crimson = '#AF0404'
coral = '#FF4848'
magenta = '#C3027D'
orange = '#ED5A01'
green = '#0A8600'
light_green = '#11C503'
teal = '#00A091'
cyan = '#00d0f0'
blue = '#0066ff'
light_blue = '#00C2F2'
dark_blue = '#004ab8'
purple = '#6D04C4'
lilac = '#EB89FF'
plum = '#862388'
pink = '#E40ACA'
baby_pink = '#FF89FD'
fuchsia = '#E102B5'
grey = '#969696'

# Obtain the figure size in inches
x_size, y_size = styledict['figure.figsize']

# Formatting for any arrows (e.g. representing upper/lower limits)
ax_frac = 1/40.         # Fraction of y-axis a vertical arrow should occupy
al = ax_frac * y_size   # Length of each arrow in inches
scale = 1./al           # 'Scale' parameter, defines length of arrows in quiver
scale_units = 'inches'  # Units of the scale parameter
aw = 0.0175 * al        # Width of each arrow shaft in inches
hw = 4.                 # Width of arrowheads in units of shaft width
hl = 3.                 # Length of arrowheads in units of shaft width
hal = 2.5               # Length of arrowheads at point of intersection w shaft

# Put all of the above arrow settings into a dictionary
arrow_settings = {
    'scale': scale,
    'scale_units': scale_units,
    'width': aw,
    'headwidth': hw,
    'headlength': hl,
    'headaxislength': hal
}

# Default markers to cycle through when plotting multiple datasets
cycle_mkr = ['o', 's', '^', 'v', '*', (8, 1, 0), 'x', '<', '>', 'p']
# Default colours to cycle through when plotting multiple datasets
cycle_clr = [dark_blue, teal, dark_red, plum, lilac, blue, green,
             orange, red, magenta]


###################
#    FUNCTIONS    #
###################

def scale_RGB_colour(rgb, scale_l=1., scale_s=1.):
    '''
    Takes any RGB code and scales its 'lightness' and saturation (according to
    the hls colour model) by factors of scale_l and scale_s.

    Parameters
    ----------
    rgb: array-like
        The RGB colour specifications.

    scale_l: float
        The factor by which to scale the lightness.

    scale_s: float
        The factor by which to scale the saturation.

    Returns
    -------
    rgb: array-like
        The RGB colour specifications of the scaled colour.
    '''
    import colorsys
    # Convert the rgb to hls (hue, lightness, saturation)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # Scale lightness and saturation and ensure the results are between 0 and 1
    l_new = max(0, min(1, l * scale_l))
    s_new = max(0, min(1, s * scale_s))
    # Convert back to rgb and return the result
    return colorsys.hls_to_rgb(h, l_new, s_new)


def scale_HEX_colour(c_hex, scale_l=1., scale_s=1.):
    '''
    Takes any colour HEX code and scales its 'lightness' and saturation
    (according to the hls colour model) by factors of scale_l and scale_s.
        c_hex: The HEX colour code.
        scale_l: The factor by which to scale the lightness.
        scale_s: The factor by which to scale the saturation.
    '''
    import matplotlib as mpl
    # Convert the colour HEX code to RGB format
    rgb = mpl.colors.ColorConverter.to_rgb(c_hex)
    # Convert back to rgb and return the result
    return scale_RGB_colour(rgb, scale_l, scale_s)


def x_offset(i, delta, log=False):
    '''
    For use when plotting multiple sets of data on a single set of axes.
    Computes an offset in the x-direction depending on which the current
    iteration of a loop through the different datasets.

    Parameters
    ----------
    i: int
        The current iteration of a loop through the different datasets.

    delta: float
        Determines the distance between each data point. If the x-axis has a
        linear scale, the offset will be some multiple of delta. If it has a
        log scale, the offset will be delta raised to some integer power.

    log: bool
        Whether the x-axis has a logged scale.

    Returns
    -------
    offset: float
        The offset along the x-axis.
    '''

    if log:
        offset = delta ** (((i + 1) // 2) * ((-1) ** (i+1)))
    else:
        offset = delta * ((i + 1) // 2) * ((-1) ** (i+1))

    return offset


def plot_correlation_matrix(S, **kwargs):
    '''
    Given a Sacc object, computes the correlation matrix from the covariance
    matrix and creates a figure displaying it.

    Parameters
    ----------
    S: sacc.sacc.Sacc object or str
        The Sacc object containing the covariance matrix to be converted into
        a correlation matrix. If a string, must be the path to a Sacc file.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure in which the correlation matrix is displayed.

    ax: matplotlib.axes._axes.Axes
        Axes on which the correlation matrix is displayed.
    '''

    import matplotlib.pyplot as plt
    import sacc
    import numpy as np

    plt.style.use(styledict)
    # Check if input is a string
    if type(S) is str:
        S = sacc.Sacc.load_fits(S)
    # Retrieve the covariance matrix
    cov = S.covariance.covmat
    # Convert to correlation matrix
    Dinv = np.diag((1. / np.sqrt(np.diag(cov))))
    corr = Dinv @ cov @ Dinv

    # Display the correlation matrix on a figure
    f, ax = plt.subplots()
    im = ax.imshow(corr, origin='upper', **kwargs)
    cbar = f.colorbar(im, ax=ax)
    cbar.set_label(r'$\rho$')

    # Adjust the axis labels so that they display the power spectra pairings
    # (assumes the last character in each C_ell label describes the bin)
    pairs = [f'{i[-1]}{j[-1]}' for i, j in S.get_tracer_combinations()]
    npairs = len(pairs)
    # Determine the width of each sub-grid in the correlation matrix
    wsg = corr.shape[0] / npairs
    # Tick positions
    tick_pos = np.arange(0.5 * wsg, (npairs+0.5)*wsg, wsg)
    # Set the positions and labels
    ax.set_xticks(tick_pos, labels=pairs)
    ax.set_yticks(tick_pos, labels=pairs)
    ax.minorticks_off()
    ax.tick_params(direction='out')

    # Return the figure and axes
    return f, ax


def plot_map(mp, field, vals_unseen=None, unseen_thresh=None, title='',
             **kwargs):
    '''
    Given a map (HealPIX or HealSparse format), will return a figure displaying
    that map.

    Parameters
    ----------
    mp: numpy.ndarray or healsparse.HealSparseMap.HealSparseMap
        The map to be displayed.

    field: str
        name of the HSC field being mapped. Must be either "hectomap",
        "spring" or "autumn".

    vals_unseen: list or None
        List of values to be treated as equivalent to hp.UNSEEN for display
        purposes. If None, all pixels will be displayed with their exact value.

    unseen_thresh: float or None
        Value below which pixels will be counted as UNSEEN. If None, all
        pixels will be displayed with their exact value.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure in which the map is displayed.
    '''
    import healpy as hp
    import healsparse as hsp
    import matplotlib.pyplot as plt
    plt.style.use(styledict)

    # Approximate field centers and corresponding suitable image dimensions
    if field == 'hectomap':
        ra_mean = 231.71
        dec_mean = 43.34
        xsize = 500
        ysize = 100
    elif field == 'spring':
        ra_mean = 5
        dec_mean = 0
        xsize = 1500
        ysize = 300
    elif field == 'autumn':
        ra_mean = 177.16
        dec_mean = 1.05
        xsize = 2500
        ysize = 300
    else:
        raise ValueError("Argument 'field' must be one of "
                         "{'hectomap', 'spring', 'autumn'}.")

    # Get the resolution of the map
    if type(mp) is hsp.HealSparseMap:
        nside = mp.nside_sparse
        mp = mp.generate_healpix_map(nest=False)
    else:
        nside = hp.npix2nside(len(mp))
    reso = hp.nside2resol(nside, arcmin=True)
    # Determine the xsize and ysize of the figure based on the NSIDE
    scale_factor = nside // 1024
    xsize *= scale_factor
    ysize *= scale_factor
    # Make a copy of the input map to avoid editing in place
    mp = mp.copy()

    # Create the figure
    fig = plt.figure(figsize=(18, 5))
    if vals_unseen is not None:
        mp = mp.copy()
        for val in vals_unseen:
            mp[mp == val] = hp.UNSEEN
    if unseen_thresh is not None:
        mp = mp.copy()
        mp[mp <= unseen_thresh] = hp.UNSEEN
    hp.gnomview(mp, rot=[ra_mean, dec_mean, 0], xsize=xsize, ysize=ysize,
                reso=reso, notext=True, fig=fig, title=title, **kwargs)


def setup_cl_plot(nbins=1, auto_only=False, label_subplots=False,
                  xlabel=None, ylabel=None):
    '''
    Sets up a multi-panel figure for displaying angular
    power spectra for different tomographic bin pairings.

    Parameters
    ----------
    nbins: int
        Number of tomographic bins (not pairings) being considered.

    auto_only: bool
        If True, will only set up plot for autocorrelation power
        spectra. Otherwise, will set up for all bin pairings.

    label_subplots: bool
        If True, will label each subplot with the corresponding
        bin pairing.

    xlabel: str or None
        Custom label for the x-axis if the independent variable is anything
        other than the angular multipoles (ells). If None, will use a default
        label.

    ylabel: str or None
        Custom label for the y-axis if the dependent variable is anything
        other than the angular power spectra (C_ells). If None, will use a
        default label.

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        Figure object.

    axes: numpy.ndarray[matplotlib.pyplot.Axes]
        Axes for each subplot.
    '''

    import cell_utils as cu
    import matplotlib.pyplot as plt

    # Get the bin pairings from the specified number of bins
    pairings, pairings_s = cu.get_bin_pairings(nbins, auto_only)

    # Define the figure size based on the number of pairings
    if auto_only:
        ncols = 2
        nrows = (nbins // 2) + (nbins % 2)
    else:
        ncols = nrows = nbins
    xsize = ncols * x_size
    ysize = nrows * y_size

    # Default x-axis label if none provided
    if not xlabel:
        xlabel = r'$\ell$'
    # Default y-axis label if none provided
    if not ylabel:
        ylabel = r'$C_{\ell}$'

    # Set up a figure for the power spectra from each redshift bin
    fig = plt.figure(figsize=(xsize, ysize))
    gs = fig.add_gridspec(ncols=ncols, nrows=nrows)

    # List to which axes will be appended
    axes = []
    for ip, (p, p_str) in enumerate(zip(pairings, pairings_s)):
        if auto_only:
            i = ip % ncols
            j = ip // ncols
        else:
            i, j = p
        # Add subplot to gridspec
        ax = fig.add_subplot(gs[j, i])
        # Only label axes if on outer edge of figure
        if j == (nrows-1):
            ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(ylabel)
        # Set loglog scale
        ax.set_xscale('log')
        ax.set_yscale('log')

        if label_subplots:
            # Add text to top-right corner to indicate which bins have
            # been compared
            ax.text(0.95, 0.95, f'({p_str})', transform=ax.transAxes,
                    ha='right', va='top', fontsize=20.)
        axes.append(ax)
    return fig, axes


def plot_cells(ax, ells, cells, err_cells=None, binned=True, color='k',
               marker='o', linestyle='-', linestyle_nve='--', label=None,
               **kwargs):
    '''
    Convenience function for plotting C_ells on an existing set	of axes.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        Axes on which the data are to be plotted.

    ells: numpy.ndarray
        NumPy array containing the effective multipoles of each bandpower.

    cells: numpy.ndarray
        NumPy array containing the angular power spectrum values.
        Must have same dimensions as ells.

    err_cells: numpy.ndarray or None
        NumPy array containing the uncertanites on the angular power spectrum
        values. Must have same dimensions as ells (assumes symmetric errors).

    binned: bool
        Whether or not the C_ells are binned into bandpowers.

    color: str
        String code for the desired colour of the datapoints.

    marker: str
        String code for the desired marker of the datapoints.

    linestyle: str
        String code for the desired linestyle (only used if C_ells are not
        binned).

    label: str
        Legend label for the data.
    '''

    if binned:
        # Determine which data are positive and which are negative
        mask_pve = cells >= 0
        mask_nve = ~mask_pve
    else:
        # Unbinned data requires different treatment
        import numpy as np
        mask_pve = np.ones_like(cells, dtype=bool)
        mask_nve = mask_pve
    # Split data into two subsets
    ells_pve = ells[mask_pve]
    ells_nve = ells[mask_nve]
    cells_pve = cells[mask_pve]
    cells_nve = cells[mask_nve]

    # Need to handle err_cells separately in case none provided
    if err_cells is not None:
        err_cells_pve = err_cells[mask_pve]
        err_cells_nve = err_cells[mask_nve]
    else:
        err_cells_pve = err_cells_nve = None

    # If C_ells are binned into bandpowers, remove the linestyle
    if binned:
        linestyle = 'none'
        linestyle_nve = 'none'
    # If they are not binned, remove the markers
    else:
        marker = 'none'

    # Plot the positive data
    cell_plot = ax.errorbar(ells_pve,
                            cells_pve,
                            yerr=err_cells_pve,
                            color=color,
                            ecolor=color,
                            marker=marker,
                            linestyle=linestyle,
                            label=label,
                            **kwargs
                            )
    # Plot the negative data
    ax.errorbar(ells_nve,
                -cells_nve,
                yerr=err_cells_nve,
                color=color,
                mec=color,
                ecolor=color,
                marker=marker,
                linestyle=linestyle_nve,
                mfc='none',
                **kwargs
                )

    return cell_plot, label
