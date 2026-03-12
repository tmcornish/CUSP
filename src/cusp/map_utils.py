##############################################################################
# Functions for the creation of HealPIX/HealSparse maps.
##############################################################################

import numpy as np
import healpy as hp
import healsparse as hsp
from output_utils import error_message


def initialiseRecMap(nside_cover, nside_sparse, labels, pixels=None, ra=None,
                     dec=None, dtypes='f8', primary=None, return_pix=True,
                     return_unique=True):
    '''
    Initialises a RecArray of HealSparse maps. Useful e.g. when a single
    quantity is measured for multiple bands.

    Parameters
    ----------
    nside_cover: int
        Resolution of the wider HealSparse map where no data exist.

    nside_sparse: int
        Resolution of the regions of the HealSparse map in which data exist.

    pixels: array-like
        List/array containing pixels (NESTED ordering) to be initialised. If
        not provided, ra and dec must be provided instead.

    ra: array-like
        RA coordinates at which data exist for the map. If not provided (along
        with dec) pixels must be provided instead.

    dec: array-like
        Dec. coordinates at which data exist for the map. If not provided
        (along with ra) pixels must be provided instead.

    labels: list
        List of strings to use as labels for each map.

    dtypes: str or type or list
        Describes the type of data with which each map will be filled. If a
        list, must have the same length of labels; otherwise, the same dtype
        will be used for all maps.

    primary: str or None
        Label of the HealSparse map to be used as the primary map.
        If None, uses the first entry of labels.

    return_pix: bool
        Return the pixel positions corresponding to the provided coordinates.

    return_unique: bool
        Return the unique pixel IDs for which data exist.

    Returns
    -------
    all_maps: recarray
        Recarray in which each entry is a HealSparse map, initialised with
        zeros at the pixel positions in which data exist, based on the
        coordinates provided.

    px_data: array
        Pixel positions corresponding to the provided coordinates.

    px_data_u: array
        Unique pixel IDs for which data exist.
    '''

    # Exported HealSparse map will contain dust maps in each band;
    # set up relevant array
    if hasattr(dtypes, '__len__') and not isinstance(dtypes, str):
        dtype_maps = [(lb, d) for lb, d in zip(labels, dtypes)]
    else:
        dtype_maps = [(lb, dtypes) for lb in labels]
    if primary is None:
        primary = labels[0]
    all_maps = hsp.HealSparseMap.make_empty(nside_cover, nside_sparse,
                                            dtype=dtype_maps, primary=primary)

    if (pixels is None) and (ra is not None) and (dec is not None):
        # Identify pixels in HealSparse map that contain sources from catalogue
        pixels = hp.ang2pix(nside_sparse,
                            np.radians(90.-dec),
                            np.radians(ra),
                            nest=True)
    elif (pixels is None) and (ra is None) and (dec is None):
        error_message('initialiseRecMap', 'Must provide either pixel IDs, or'
                      ' RA and Dec. coordinates.')

    # Determine the unique pixels
    pixels_u = np.unique(pixels)

    # Initially fill these pixels with zeroes so that HealSparse can update
    # their values later
    all_maps.update_values_pix(pixels_u,
                               np.zeros(len(pixels_u), dtype=dtype_maps))

    return all_maps, pixels, pixels_u


def pixelCountsFromCoords(ra, dec, nside_cover, nside_sparse,
                          return_pix_and_vals=False):
    '''
    Given sets of coordinates (RA and Dec.), counts the number of objects in
    each pixel of a HealPIX map with the specified NSIDE.

    Parameters
    ----------
    ra: array-like
        RA coordinates of each object.

    dec: array-like
        Dec. coordinates of each object.

    nside_cover: int
        Resolution of the wider HealSparse map where no data exist.

    nside_sparse: int
        Resolution of the regions of the HealSparse map in which data exist.

    return_pix_and_vals: bool
        If True, also returns the IDs and corresponding values for occupied
        pixels.

    Returns
    -------
    counts_map: HealSparseMap
        HealSparse map containing the number of sources in each pixel.
    '''
    # Initialise a HealSparse integer map
    counts_map = hsp.HealSparseMap.make_empty(nside_cover,
                                              nside_sparse,
                                              np.int32)
    # Convert the provided coordinates into pixel IDs
    px_data = hp.ang2pix(nside_sparse,
                         np.radians(90.-dec), np.radians(ra), nest=True)
    # Get the unique pixel IDs
    px_data_u = np.unique(px_data)

    # Count the number of sources in each pixel (going from 0 to max(px_data))
    N = np.bincount(px_data).astype(np.int32)
    # Identify all pixels containing at least one source
    good = N > 0
    # Fill the map at these positions with the number of sources in the pixel
    counts_map[px_data_u] = N[good]

    if return_pix_and_vals:
        return counts_map, px_data_u, N[good]
    else:
        return counts_map


def countsInPixels(ra, dec, nside_cover, nside_sparse, pix_ids,
                   return_vals=False):
    '''
    Given sets of coordinates (RA and Dec.), counts the number of objects in
    pixels with the provided IDs in a HealPIX map.

    Parameters
    ----------
    ra: array-like
        RA coordinates of each object.

    dec: array-like
        Dec. coordinates of each object.

    nside_cover: int
        Resolution of the wider HealSparse map where no data exist.

    nside_sparse: int
        Resolution of the regions of the HealSparse map in which data exist.

    return_pix_and_vals: bool
        If True, also returns the IDs and corresponding values for occupied
        pixels.

    Returns
    -------
    counts_map: HealSparseMap
        HealSparse map containing the number of sources in each pixel.
    '''
    # Initialise a HealSparse integer map
    counts_map = hsp.HealSparseMap.make_empty(nside_cover,
                                              nside_sparse,
                                              np.int32)
    # Convert the provided coordinates into pixel IDs
    px_data = hp.ang2pix(nside_sparse, ra, dec, nest=True, lonlat=True)
    # Make an array of weights for counting galaxies in each pixel
    weights = np.zeros_like(px_data)
    # Give specified pixels a weight of 1
    weights[np.in1d(px_data, pix_ids)] = 1

    # Count the number of sources in each pixel (going from 0 to max(px_data))
    N = np.bincount(px_data,
                    weights=weights,
                    minlength=hp.nside2npix(nside_sparse)).astype(np.int32)
    # Fill the map at these positions with the number of sources in the pixel
    counts_map[pix_ids] = N[pix_ids]

    if return_vals:
        return counts_map, N[pix_ids]
    return counts_map


def pixelMeanStd(quant, pix, remove_zeros=True):
    '''
    Calculate the mean and standard deviation of a given quantity at each
    pixel in a map.

    Parameters
    ----------
    quant: array-like
        Values of the quantity being pixelised.

    pix: array-like or None
        Pixels corresponding to the coordinates at which the quantity is
        measured.

    remove_zeros: bool
        If True, only returns results for pixels containing data.

    Returns
    -------
    qmean: array-like
        Mean value of the quantity at each pixel.

    qstd: array-like
        Standard deviation of the quantity at each pixel.
    '''

    # Count the number of values of quant associated with each pixel
    N = np.bincount(pix)
    # Calculate the sum and the sum of the squares of the quantity per pix
    qsum = np.bincount(pix, weights=quant)
    qsqsum = np.bincount(pix, weights=quant**2.)

    if remove_zeros:
        # Identify pixels in which at least one value of the quantity exists
        good = (N > 0)
        # Calculate mean and variance of the quantity at each 'good' pixel
        qmean = qsum[good] / N[good]
        qmeansq = qmean ** 2.
        qvar = (qsqsum[good] - (2 * qmean * qsum[good])) / N[good] + qmeansq
        # Identify pixels for which the variance is negative (arises from
        # rounding errors)
        nve_var = qvar < 0.
        qvar[nve_var] = 0.
        qstd = np.sqrt(qvar)
    else:
        # Suppress numpy DivideByZero warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            # Calculate mean and std of the quantity at all pixels
            qmean = qsum / N
            qstd = np.sqrt((qsqsum - (2 * qmean * qsum)) / N + (qmean ** 2.))

    return qmean, qstd


def createMeanStdMap(ra, dec, quant, nside_cover, nside_sparse):
    '''
    Creates maps containing the mean and standard deviation of a given
    quantity in each pixel.

    Parameters
    ----------
    ra: array-like
        RAs at which the flags are provided.

    dec: array-like
        Decs at which the flags are provided.

    quant: array-like or list of array-likes
        Values of the desired quantity at each position defined by the
        provided RAs and Decs.

    nside_cover: int
        NSIDE parameter definining the low-resolution regions of the map
        (where no data exist).

    nside_sparse: int
        NSIDE parameter definining the high-resolution regions of the map
        (where data exist).

    Returns
    -------
    mean_map: HealSparseMap
        Map containing the means of the specified quantity in each pixel.

    std_map: HealSparseMap
        Map containing the standard deviations of the specified quantity in
        each pixel.
    '''

    # Set up two maps with desired resolutions (one for mean and one for std)
    mean_map = hsp.HealSparseMap.make_empty(nside_cover,
                                            nside_sparse,
                                            np.float64)
    std_map = hsp.HealSparseMap.make_empty(nside_cover,
                                           nside_sparse,
                                           np.float64)

    # Convert coordinates into pixel coordinates in the high-resolution map
    px_data = hp.ang2pix(nside_sparse, ra, dec, nest=True, lonlat=True)
    px_data_u = np.unique(px_data)

    # Calculate mean and std of the quantity at each pixel and populate maps
    mean_map[px_data_u], std_map[px_data_u] = pixelMeanStd(quant,
                                                           px_data,
                                                           remove_zeros=True)

    return mean_map, std_map


def createMask(ra, dec, flags, nside_cover, nside_sparse):
    '''
    Creates a mask using a set of flags defined at the provided coordinates.

    Parameters
    ----------
    ra: array-like
        RAs at which the flags are provided.

    dec: array-like
        Decs at which the flags are provided.

    flags: array-like or list of array-likes
        Boolean arrays containing the flag at each position. Pixels containing
        any True flags will be masked.

    nside_cover: int
        NSIDE parameter definining the low-resolution regions of the map
        (where no data exist).

    nside_sparse: int
        NSIDE parameter definining the high-resolution regions of the map
        (where data exist).

    Returns
    -------
    mask: HealSparseMap
        Map containing 0s at masked positions and 1s elsewhere.
    '''

    # Begin by counting sources in each pixel, since pixels with zero sources
    # will be masked
    counts_map = pixelCountsFromCoords(ra, dec, nside_cover, nside_sparse)

    # Initialise a mask with the same NSIDE parameters as the counts map
    mask = hsp.HealSparseMap.make_empty(nside_cover, nside_sparse, np.int32)
    # Fill any occupied pixels with a 1
    mask[counts_map.valid_pixels] = 1

    # Convert the provided RAs and Decs to pixel IDs in the new-resolution mask
    px_data = hp.ang2pix(nside_sparse,
                         np.radians(90.-dec),
                         np.radians(ra),
                         nest=True)

    if type(flags) is not list:
        flags = [flags]
    # Cycle through the flags
    for flag in flags:
        # Identify where the flag=True
        px_to_mask = px_data[flag]
        px_u = np.unique(px_to_mask)
        mask[px_u] = 0

    return mask


def maskAreaSkyCoverage(mask, thresh=0.):
    '''
    Takes a HealSparseMap mask and calculates the unmasked area and the
    fractional sky coverage. Assumes that pixels with smaller values are
    `more heavily masked'.

    Parameters
    ----------
    mask: HealSparseMap
        The mask for which the unmasked area and fractional sky coverage will
        be calculated.

    thresh: float
        Threshold below which pixels will be classed as masked.

    Returns
    -------
    A: float
        Unmasked area (in square degrees).

    f_sky: float
        Fractional sky coverage.
    '''

    # Calculate the area of one pixel in square degrees
    A_pix = hp.nside2pixarea(mask.nside_sparse, degrees=True)
    # Determine the number of pixels above the mask threshold
    vpix = mask.valid_pixels
    N_hits = mask[vpix][mask[vpix] > thresh].sum()
    # Calculate the total unmasked area and fractional sky coverage
    A = N_hits * A_pix
    f_sky = A / (4. * np.pi * (180. / np.pi) ** 2.)

    return A, f_sky


def healsparseToHDF(hsp_map, fname, pix_scheme='ring', group='',
                    metadata=None):
    '''
    Takes a HealSparse map and reformats it to an HDF5 dataset containing the
    pixel IDs and values.

    Parameters
    ----------
    hsp_map: HealSparseMap
        Map to be converted from HealSparseMap to HDF5 format.

    fname: str
        Filename for the output HDF5 file.

    pix_scheme: str
        The desired pixelisation scheme of the map. Either 'ring' or 'nest'.
        Input map is assumed to be 'nest' as HealSparse maps have this scheme
        by definition.

    group: str
        Group within which the relevant data are expected to reside.

    metadata: dict or None
        Dictionary of additional metadata to be stored for the specified group.
    '''

    import h5py

    # Get the valid pixels and correpsonding values from the map
    vpix = hsp_map.valid_pixels
    values = hsp_map[vpix]
    # If output is to be in ring scheme, convert pixel IDs to ring equivalent
    if pix_scheme.lower() == 'ring':
        vpix = hp.nest2ring(hsp_map.nside_sparse, vpix)
    elif pix_scheme.lower() == 'nest':
        pass
    else:
        raise ValueError('pix_scheme must be either "nest" or "ring".')

    with h5py.File(fname, 'w') as hf:
        # Create the relevant group if it doesn't already exist
        if len(group) != 0:
            g = hf.create_group(group)
        else:
            g = hf
        # Add some metadata to describe the pixelisation
        g.attrs['pixelization'] = 'healpix'
        g.attrs['nside'] = hsp_map.nside_sparse
        # Add any additional metadata
        if metadata is not None:
            for md in metadata:
                # G.attrs = dict(g.attrs, **metadata)
                g.attrs[md] = metadata[md]
        _ = hf.create_dataset(f'{group}/pixel',
                              data=vpix,
                              dtype=vpix.dtype)
        _ = hf.create_dataset(f'{group}/value',
                              data=values,
                              dtype=values.dtype)


def healsparseToFITS(hsp_map, fname, nest=False):
    '''
    Takes a HealSparse map and reformats it to healpy FITS map.

    Parameters
    ----------
    hsp_map: HealSparseMap
        Map to be converted from HealSparseMap to FITS format.

    fname: str
        Filename for the output HDF5 file.

    nest: bool
        Whether the output map should be in 'nest' format. Input map is assumed
        to be 'nest' as HealSparse maps have this scheme by definition.
    '''

    # Convert the map to healpy format
    hp_map = hsp_map.generate_healpix_map(nest=nest)
    # Write to file
    hp.write_map(fname, hp_map, nest=nest, column_names=['VALUE'],
                 overwrite=True)


def load_map(map_path, is_healpix=False, apply_mask=False, is_systmap=False,
             mask=None):
    '''
    Loads an individual HealSparse or HEALPix map and returns their pixel
    values in HEALPix RING ordering. If told to, will also multiply the map
    by the mask and/or calculate the mean of the map and subtract it from
    all pixels. Both of these operations require the mask (in full HealPIX
    format) as input.

    Parameters
    ----------
    map_path: str
        Path to the map being read.

    is_healpix: bool (optional)
        If True, input map is in HEALPix format rather than HealSparse (False
        by default).

    apply_mask: bool
        If True, will perform element-wise multiplication by the mask (given
        as separate input).

    is_systmap: bool
        If True, subtracts the mean from the value of all pixels (necessary
        for systematics maps).

    mask: MaskData
        Object containing the mask and any potentially relevant pre-computed
        values.

    Returns
    -------
    fs_map: np.array
        Full-sky map data (RING ordering).
    '''

    if is_healpix:
        # Load the map directly and convert to RING ordering if needed
        fs_map = hp.read_map(map_path, nest=False)
    else:
        # Initialise an empty full-sky map (NOTE: can take a lot of memory for
        # high nside_sparse)
        fs_map = hsp.HealSparseMap.read(
            map_path
            ).generate_healpix_map(nest=False)
    fs_map[fs_map == hp.UNSEEN] = 0.

    if is_systmap:
        if mask is not None:
            mu = np.sum(fs_map[mask.vpix_ring]
                        * mask.mask_full[mask.vpix_ring]) / mask.sum
            fs_map[mask.vpix_ring] -= mu
        else:
            print('Could not correct systematics map; no MaskData provided.')

    if apply_mask:
        if mask is not None:
            fs_map *= mask.mask_full
        else:
            print('Could not apply mask to map; no MaskData provided.')

    return fs_map


def load_tomographic_maps(map_path, fullsky=True, apply_mask=False, mask=None,
                          idx=None):
    '''
    Loads files containing maps split into tomographic bins (e.g. delta_g) and
    (by default) returns their pixel values in healPIX RING ordering, or
    simply returns a list of each of the maps in HealSparseMap format . If
    told to, will also multiply the map by the mask, in which case a MaskData
    object is required as input.

    Parameters
    ----------
    map_path: str
        Path to the map being read.

    apply_mask: bool
        If True, will perform element-wise multiplication by the mask (given
        as separate input).

    mask: MaskData
        Object containing the mask and any potentially relevant pre-computed
        values.

    idx: int or str or list or None
        The index of the map to be read. If an int, will load the one map
        corresponding to that index. If a string, will load the one map whose
        name is that string. If a list is provided it must contain either all
        ints or strings; this function will load and return all maps with
        IDs/names matching the list entries.

    Returns
    -------
    out_maps: list
        List containing full-sky data (RING ordering) for each tomographic map.
    '''
    # Empty list in which the full-sky maps will be stored
    out_maps = []

    # Load the HealSparse file
    hsp_map = hsp.HealSparseMap.read(map_path)

    # Check the format of the idx argument, if provided
    if idx is not None:
        nd_idx = np.ndim(idx)
        if nd_idx == 0:
            if type(idx) is str:
                to_read = [idx]
            elif type(idx) is int:
                to_read = [hsp_map.dtype.names[idx]]
            else:
                error_message('load_tomographic_maps',
                              'Single values for idx must be either int '
                              'or str')
                return
        elif nd_idx == 1:
            if all([type(x) is str for x in idx]):
                to_read = idx
            elif all([type(x) is int for x in idx]):
                to_read = [hsp_map.dtype.names[x] for x in idx]
            else:
                error_message('load_tomographic_maps',
                              'Lists provided for idx must contain either all '
                              'int or all str')
                return
        else:
            error_message('load_tomographic_maps',
                          'idx must be an int, str, or list of either')
            return
    else:
        to_read = hsp_map.dtype.names

    # Cycle through the maps
    for d in to_read:

        if fullsky:
            # Create full-sky realisation of the map
            map_now = hsp_map[d].generate_healpix_map(nest=False)
            map_now[map_now == hp.UNSEEN] = 0.
        else:
            # Otherwise, just retrieve the relevant HealSparseMap
            map_now = hsp_map[d]

        # Multiply by the mask if told to do so
        if apply_mask:
            if mask is not None:
                if fullsky:
                    map_now *= mask.mask_full
                else:
                    vpix_map = map_now.valid_pixels
                    vpix_mask = mask.vpix_nest
                    vpix_diff = np.array(list(set(vpix_mask) - set(vpix_map)))
                    map_now[vpix_diff] = 0
                    map_now[vpix_mask] *= mask.mask_hsp[vpix_mask]
            else:
                print('Could not apply mask to map; no MaskData provided.')

        # Append the full-sky map to the list
        out_maps.append(map_now)

    return out_maps


class MaskData:
    '''
    Convenience class for containing important information about the survey
    mask. Upon initialisation only requires the name of the HealSparse file.
    '''

    def __init__(
            self,
            hsp_file
            ):
        # Load the HealSparse map and retrieve the high and low NSIDEs
        self.filename = hsp_file
        self.mask_hsp = hsp.HealSparseMap.read(self.filename)
        self.nside = self.mask_hsp.nside_sparse
        self.nside_cover = self.mask_hsp.nside_coverage
        # Get the valid pixels in RING and NEST ordering
        self.vpix_nest = self.mask_hsp.valid_pixels
        self.vpix_ring = hp.nest2ring(self.nside, self.vpix_nest)
        # Now make a full-sky version (RING), replacing any UNSEENs with 0
        self.mask_full = self.mask_hsp.generate_healpix_map(nest=False)
        self.mask_full[self.mask_full < 0.] = 0.

        # Get the RA and Dec. of each unmasked pixel
        self.ra_vpix, self.dec_vpix = hp.pix2ang(self.nside,
                                                 self.vpix_ring,
                                                 lonlat=True)

        # Compute the sum, mean, and mean squared of the mask
        self.sum = np.sum(self.mask_full)
        self.mean = np.mean(self.mask_full)
        self.meansq = np.mean(self.mask_full ** 2.)
