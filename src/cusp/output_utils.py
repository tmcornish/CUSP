###############################################################################
# Module containing functions for formatting outputs from scripts.
###############################################################################

def error_message(module, message):
    '''
    Prints a nicely formatted error message. For use within other modules as a
    means of identifying issues.

    Parameters
    ----------
    module: str
        The name of the module being debugged.

    message: str
        The body of the error message to print.
    '''
    err_str = [
        colour_string(f'{module}\n', 'cyan'),
        colour_string('Error: '),
        colour_string(message, 'white')]
    print(''.join(err_str))


def colour_string(s, c='red'):
    '''
    Reformats a string so that it can be printed to the Terminal in colour
    (against a black background).

    Parameters
    ----------
    s: str
        The string to be printed in colour.

    c: str
        The desired colour (must be one of the seven available choices; see
        below).

    Returns
    -------
    s_new: str
        The input string, reformatted to be displayed with the desired colour
        in the Terminal.
    '''

    # List of possible colour symbols
    colours = ['red', 'green', 'orange', 'blue', 'purple',
               'cyan', 'white']
    # Corresponding codes
    codes = ['0;31;40', '0;32;40', '0;33;40', '0;34;40', '0;35;40',
             '0;36;40', '0;37;40']
    # Identify the code corresponding to the colour selected in the argument
    try:
        code_sel = codes[colours.index(c)]
        # Use the relevant code to ensure the string is printed to the
        # terminal in the chosen colour
        s_new = '\x1b[%sm%s\x1b[0m' % (code_sel, s)
        # Return the new string
        return s_new
    except ValueError:
        # If the user did not select an available colour, print an explanation
        # and return the original string
        error_message('colour_string',
                      'Selected colour not available. Available colours are:'
                      '\nred\ngreen\norange\nblue\npurple\nolive\nwhite\n')
        return s


def string_important(s):
    '''
    Prints the provided string, using large numbers of '#'s to make it easy to
    spot when running the code.

    Parameters
    ----------
    s: str
        String to print.

    Returns
    -------
    Input string reformatted to be easily visible in the Terminal.
    '''
    # Formatting
    N_str = len(s)
    N_pad = 8
    N_total = N_str + 2 * (N_pad + 1)
    pad_newline = '#' * N_total
    pad_textline = '#' * N_pad
    # Line containing the important text with padding
    textline = ' '.join([pad_textline, s, pad_textline])
    return '\n'.join([pad_newline, textline, pad_newline])


def array_to_fits(data, filename, CTYPE1='RA', CTYPE2='DEC',
                  CRPIX=[1, 1], CRVAL=[0, 0], CDELT=[1, 1]):
    '''
    Takes an array and details describing a coordinate system and creates a
    FITS file.

    Parameters
    ----------
    data: array
        The array containing the data.

    filename: str
        Output file name.

    CTYPE1: str
        Name of the variable along axis 1.

    CTYPE2: str
        Name of the variable along axis 2.

    CRPIX: array-like
        Reference pixels in [X,Y].

    CRVAL: array-like
        Reference pixel values in [X,Y].

    CDELT: array-like
        Pixel scales in [X,Y].
    '''
    from astropy.io import fits

    # Create a PrimaryHDU from the chi^2 grid
    hdu = fits.PrimaryHDU(data)
    # Update the relevant parameters in the header
    hdu.header['CTYPE1'] = CTYPE1
    hdu.header['CTYPE2'] = CTYPE2
    hdu.header['CRPIX1'] = CRPIX[0]
    hdu.header['CRPIX2'] = CRPIX[1]
    hdu.header['CRVAL1'] = CRVAL[0]
    hdu.header['CRVAL2'] = CRVAL[1]
    hdu.header['CDELT1'] = CDELT[0]
    hdu.header['CDELT2'] = CDELT[1]
    # Create an HDUList
    hdul = fits.HDUList([hdu])
    # Write this to the file
    hdul.writeto(filename, overwrite=True)


def write_output(t, fname, **kwargs):
    '''
    Writes an astropy Table to a file (type inferred from filename) with
    appropriate metadata.

    Parameters
    ----------
    t: astropy.table.Table
        Input catalogue.

    fname: str
        Filename to be given to the output file.
    '''
    from astropy.io import fits

    # Set up the header
    hdr = fits.Header()
    for key in kwargs:
        hdr[key.upper()] = kwargs[key]
    prm_hdu = fits.PrimaryHDU(header=hdr)

    # Convert the catalogue into an HDU
    cat_hdu = fits.table_to_hdu(t)

    # Write to file
    hdul = fits.HDUList([prm_hdu, cat_hdu])
    hdul.writeto(fname, overwrite=True)


def write_output_hdf(t, fname, colnames=None, group=None, mode='a'):
    '''
    Writes an astropy Table to a hdf5 file.

    Parameters
    ----------
    t: astropy.table.Table
        Input catalogue.

    fname: str
        Filename to be given to the output file. If the file already exists,
        will try to append to existing data in the file.

    colnames: list or None
        List of columns to be included in the output file. If None, writes all
        columns to the output.

    group: str
        Name of the group to which the data will be written. If None, will
        write the data to the 'root' of the hierarchy in the output file.

    mode: str
        Mode in which to open the HDF file (e.g. 'w' for 'write', 'r' for
        'read', etc.)
    '''
    import h5py

    # If colnames=None, set equal to the list of column names in t
    if colnames is None:
        colnames = t.colnames
    # If group=None, assign an empty string
    if group is None:
        group = ''

    # Get the length of the table
    N = len(t)
    # If table has zero entries, skip  remaining steps or an error will occur
    if N == 0:
        return
    # Open the file
    with h5py.File(fname, mode) as hf:
        # Cycle through the columns
        for col in colnames:
            # Get the dtype of the column data
            dt = t[col].dtype
            # Create the relevant group if it doesn't already exist
            if len(group) != 0:
                _ = hf.require_group(group)
            # See if the current column already exists as a DataSet
            if col in hf[f'/{group}'].keys():
                # Reshape existing data and fill empty entries with new data
                dset = hf[f'/{group}/{col}']
                dset.resize((len(dset)+N,))
                dset[-N:] = t[col]
            else:
                # Create the dataset if it doesn't exist
                dset = hf.create_dataset(f'{group}/{col}',
                                         shape=(N,),
                                         data=t[col],
                                         maxshape=(None,),
                                         dtype=dt)


def h5py_dataset_iterator(g, prefix=''):
    import h5py
    for key, item in g.items():
        path = '{}/{}'.format(prefix, key)
        if isinstance(item, h5py.Dataset):  # Test for dataset
            yield (path, item)
        elif isinstance(item, h5py.Group):  # Test for group (go down)
            yield from h5py_dataset_iterator(item, path)
