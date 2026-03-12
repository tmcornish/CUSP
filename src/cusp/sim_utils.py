##############################################################################
# Functions and classes for creating and analysing simulated data.
# TODO: documentation
# TODO: add methods of catching errors and printing possible solutions
##############################################################################

import healpy as hp
import numpy as np
import pymaster as nmt


class GaussianSim(object):
    '''
    Class for generating a Gaussian realisation of a map generated from some
    input angular power spectrum. Also includes functions for applying a mask
    and contaminating with templates, if provided.

    Parameters
    ----------
    cl: array[float]
        Input angular power spectrum, assumed to be defined at every multipole
        from 0 to (at least) 3 * nside - 1.

    nside: int
        HealPIX Nside value; determines resolution of the map.

    mask: array[float] or None
        Map defining the survey geometry. Assumed to have the same resolution
        and format as the map to be created.

    templates: array[float] or None
        Array containing any systematics templates with which the map is to be
        contaminated. Must have shape of either (Nsyst, 1, Npix) or
        (Nsyst, Npix), where Nsyst is the number of templates and Npix is the
        number of pixels in a map with the desired Nside.

    seed: int or None
        Seed to use when randomly generating the map.
    '''
    def __init__(
            self,
            cl,
            nside,
            mask=None,
            templates=None,
            seed=None
    ):
        # Initialise properties
        self.cl_in = cl
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.lmax = 3 * nside - 1
        self.is_masked = False
        self.is_contaminated = False
        self.seed = seed
        # Maps
        if mask is None:
            self.mask = np.ones(self.npix)
        else:
            self.mask = mask
        self.fsky = np.mean(self.mask ** 2)
        # Templates
        self.templates = templates
        if templates is not None:
            self.ntemp = len(templates)
        # Generate a Gaussian realisation of the map
        np.random.seed(self.seed)
        self.map = hp.synfast(cl, nside)

    def apply_mask(self):
        '''
        Applies the survey mask to the generated map.
        '''
        self.map *= self.mask
        self.is_masked = True

    def contaminate_map(self, alphas):
        '''
        Contaminates the generated map with the provided templates, using
        the specified contamination amplitudes.

        Parameters
        ----------
        alphas: array[float]
            Contamination amplitudes for each template, i.e. for a given
            template t with amplitude alpha, the map will be contaminated
            by adding alpha*t.
        '''
        if self.templates is None:
            raise ValueError(
                'Templates must be provided upon initialisation in order'
                'to contaminate the map.'
            )
        if self.is_masked:
            self.templates *= self.mask
        # Sum of contributions form all systematics
        self.syst = np.sum(self.templates * alphas[:, None, None], axis=0)[0]
        # Contaminate map
        self.map += self.syst
        self.is_contaminated = True
        self.alphas = alphas

    def make_nmtfield(self, deproject=False):
        '''
        Makes an NmtField object from the class properties, applying linear
        template deprojection if requested.

        Parameters
        ----------
        deproject: bool
            Whether or not to apply linear template deprojection.
        '''
        if deproject:
            temps = self.templates
        else:
            temps = None

        df = nmt.NmtField(
            self.mask,
            [self.map],
            templates=temps,
            masked_on_input=self.is_masked
        )
        self.field = df

    @classmethod
    def run_analysis_map(
        cls,
        cl,
        nside,
        mask,
        templates,
        alphas,
        wsp=None,
        nmtbin=None,
        cls_to_compute=['all'],
        compute_db_true=False,
        compute_db_guess=False,
        seed=None
    ):
        '''
        Method for generating a Gaussian sim and computing angular power
        spectra for four scenarios:
            1. Map is not contaminated, no deprojection applied (ncnd)
            2. Map is not contaminated, but deprojection applied (ncd)
            3. Map is contaminated, but no deprojection applied (cnd)
            4. Map is contaminated, deprojection applied (cd).

        Parameters
        ----------
        cl: array[float]
            Input angular power spectrum, assumed to be defined at every
            multipole from 0 to (at least) 3 * nside - 1.

        nside: int
            HealPIX Nside value; determines resolution of the map.

        mask: array[float]
            Map defining the survey geometry. Assumed to have the same
            resolution and format as the map to be created.

        templates: array[float]
            Array containing any systematics templates with which the map is
            to be contaminated. Must have shape of either (Nsyst, 1, Npix) or
            (Nsyst, Npix), where Nsyst is the number of templates and Npix is
            the number of pixels in a map with the desired Nside.

        alphas: array[float]
            Contamination amplitudes for each template, i.e. for a given
            template t with amplitude alpha, the map will be contaminated
            by adding alpha*t.

        wsp: NmtWorkspace or None
            Workspace with stored mode-coupling matrix for use in decoupling
            computed pseudo-C_ells. If not provided, will construct one in
            situ.

        nmtbin: NmtBin or None
            Object defining the bandpowers to use for decoupled C_ells.
            Ignored if wsp is not None. Cannot be None if wsp is None.

        cls_to_compute: list
            List of strings corresponding to the C_ells one wishes to compute.
            These strings must be in {'ncnd', 'ncd', 'cnd', 'cd', 'all'}. If
            'all', will compute all of the above options.

        compute_db_true: bool
            Whether or not to compute deprojection bias using the input C_ell
            as an estimate of the true C_ell.

        compute_db_guess: bool
            Whether or not to compute deprojection bias using the measured
            C_ell divided by the sky fraction as an estimate of the true C_ell.

        seed: int or None
            Seed to use when generating the map.
        '''
        # Determine which C_ells to compute
        if 'all' in cls_to_compute:
            cls_to_compute = ['ncnd', 'ncd', 'cnd', 'cd']
        # Initialise simulation
        gsim = cls(cl, nside, mask, templates, seed)
        # Set up dictionary to store results
        gsim.analysis = {}

        # No contamination, no deprojction (ncnd)
        if 'ncnd' in cls_to_compute:
            gsim.make_nmtfield(deproject=False)
            # Construct NmtWorkspace if none provided
            if wsp is None:
                wsp = nmt.NmtWorkspace.from_fields(
                    gsim.field, gsim.field, nmtbin
                )
            gsim.analysis['pcl_ncnd'] = nmt.compute_coupled_cell(
                gsim.field, gsim.field
            )
            gsim.analysis['cl_ncnd'] = wsp.decouple_cell(
                gsim.analysis['pcl_ncnd']
            )

        # No contamination, deprojection (ncd)
        if 'ncd' in cls_to_compute:
            gsim.make_nmtfield(deproject=True)
            if wsp is None:
                wsp = nmt.NmtWorkspace.from_fields(
                    gsim.field, gsim.field, nmtbin
                )
            gsim.analysis['pcl_ncd'] = nmt.compute_coupled_cell(
                gsim.field, gsim.field
            )
            gsim.analysis['cl_ncd'] = wsp.decouple_cell(
                gsim.analysis['pcl_ncd']
            )

        if not any(['cnd' in cls_to_compute, 'cd' in cls_to_compute]):
            return gsim

        gsim.contaminate_map(alphas=alphas)

        # Contamination, no deprojection (cnd)
        if 'cnd' in cls_to_compute:
            gsim.make_nmtfield(deproject=False)
            if wsp is None:
                wsp = nmt.NmtWorkspace.from_fields(
                    gsim.field, gsim.field, nmtbin
                )
            gsim.analysis['pcl_cnd'] = nmt.compute_coupled_cell(
                gsim.field, gsim.field
            )
            gsim.analysis['cl_cnd'] = wsp.decouple_cell(
                gsim.analysis['pcl_cnd']
            )

        # Contamination, deprojection (cd)
        if 'cd' in cls_to_compute:
            gsim.make_nmtfield(deproject=True)
            if wsp is None:
                wsp = nmt.NmtWorkspace.from_fields(
                    gsim.field, gsim.field, nmtbin
                )
            gsim.analysis['pcl_cd'] = nmt.compute_coupled_cell(
                gsim.field, gsim.field
            )
            gsim.analysis['cl_cd'] = wsp.decouple_cell(
                gsim.analysis['pcl_cd']
            )
            # Deprojection bias
            if compute_db_true:
                gsim.analysis['clb_true'] = wsp.decouple_cell(
                    nmt.deprojection_bias(
                        gsim.field,
                        gsim.field,
                        cl.reshape(1, -1)
                    )
                )
            if compute_db_guess:
                gsim.analysis['clb_guess'] = wsp.decouple_cell(
                    nmt.deprojection_bias(
                        gsim.field,
                        gsim.field,
                        gsim.analysis['pcl_cd'] / gsim.fsky
                    )
                )
            gsim.analysis['alphas'] = gsim.field.alphas
        return gsim


class PoissonSim(GaussianSim):
    '''
    Class for generating a Poisson realisation of a catalogue given an
    input angular power spectrum. Note that the catalogue is not generated
    on instantiation; make_catalogue() must be called in order to generate
    the catalogue.

    Parameters
    ----------
    cl: array[float]
        Input angular power spectrum, assumed to be defined at every multipole
        from 0 to (at least) 3 * nside - 1.

    nside: int
        HealPIX Nside value; determines resolution of the map.

    ndata: int
        The expectation value of the Poisson distribution from which the
        number of simulated sources will be drawn.

    mask: array[float] or None
        Map defining the survey geometry. Assumed to have the same resolution
        and format as the map to be created.

    templates: array[float] or None
        Array containing any systematics templates with which the map is to be
        contaminated. Must have shape of either (Nsyst, 1, Npix) or
        (Nsyst, Npix), where Nsyst is the number of templates and Npix is the
        number of pixels in a map with the desired Nside.

    pos_ran: array[float] or None
        Array containing RAs and Decs for a set of randoms.

    templates_ran: array[float] or None
        Array conaining the values of each template (if provided) evaluated at
        the position of each random. If None but templates are still provided,
        templates_ran can be generated in situ if required.

    lmax_deproj: int or None
        Maximum multipole to which contaminants will be deprojected. If None,
        will default to the same lmax as is used for computing the C_ells
        generally (3 * N_side - 1).

    ntries: int
        Number of tries allowed to generate a masked map whose pixels are all
        >= -1 and can thus be Poisson sampled everywhere. If such a map has not
        been successfully generated after this many attempts, will print a
        warning about proceeding with Poisson sampling.

    seed: int or None
        Seed to use when generating the map and catalogue.
    '''
    def __init__(
            self,
            cl,
            nside,
            ndata,
            mask=None,
            templates=None,
            pos_ran=None,
            templates_ran=None,
            lmax_deproj=None,
            ntries=10,
            seed=None
    ):
        # Initialise Gaussian sim
        GaussianSim.__init__(self, cl, nside, mask, templates, seed)
        # Ensure no pixels have values <-1 (cannot be Poisson sampled)
        self.success = False
        for _ in range(ntries):
            if ((self.map * self.mask) >= -1).all():
                self.success = True
                break
            self.map = hp.synfast(cl, nside)
        if not self.success:
            print('WARNING: pixels present in map with values < -1 even after '
                  f'{ntries} attempts. Proceed with caution when Poisson '
                  'sampling.')
        # Initialise additional attributes
        self.mu_poisson = ndata
        self.ndata = None
        self.pos_ran = pos_ran
        self.templates_ran = templates_ran
        if lmax_deproj is None:
            self.lmax_deproj = self.lmax
        else:
            self.lmax_deproj = lmax_deproj
        # Randoms-related attributes
        if self.pos_ran is not None:
            self.w_ran = np.ones_like(pos_ran[0])
            self.ipix_ran = hp.ang2pix(nside, *pos_ran, lonlat=True)
        else:
            self.w_ran = None
            self.ipix_ran = None

    def make_catalogue(self, use_mean=False):
        '''
        Generates a realisation of a catalogue from the generated map, with
        the number of sources drawn from a Poisson distribution with the
        specified expectation value.
        '''
        # Use mu_poissn directly or draw Poisson number
        if use_mean:
            nwant = self.mu_poisson
            seeded = False
        else:
            np.random.seed(self.seed)
            seeded = True
            nwant = np.random.poisson(self.mu_poisson)
        self.ndata = nwant
        if self.is_masked:
            pmap = self.mask + self.map
        else:
            pmap = self.mask * (1 + self.map)
        pmap /= np.amax(pmap)

        # Calculate number of objects to generate
        pmean = np.mean(pmap)
        ntot = int((self.mu_poisson+5*np.sqrt(self.mu_poisson))/pmean)
        # Determine which to keep
        if not seeded:
            np.random.seed(self.seed)
        th = np.arccos(-1+2*np.random.rand(ntot))
        phi = 2*np.pi*np.random.rand(ntot)
        u = np.random.rand(ntot)
        ipix = hp.ang2pix(self.nside, th, phi)
        p = pmap[ipix]
        keep = u <= p
        # Truncate
        phi = phi[keep][:nwant]
        th = th[keep][:nwant]
        ipix = ipix[keep][:nwant]
        pos = np.array([np.degrees(phi), 90-np.degrees(th)])
        # Store source positions and weights (assumed to be ones for now)
        self.pos_data = pos
        self.w_data = np.ones_like(ipix)
        self.ipix_data = ipix

    def make_deltag_map(self):
        '''
        Constructs a map of the galaxy density contrast from the positions in
        the simulated catalogue. Also computes and stores the shot noise
        associated with the catalogue. Note that make_catalogue() must be run
        prior to calling this method.
        '''
        # Create map of galaxy counts
        try:
            nmap = np.bincount(self.ipix_data, minlength=self.npix)
        except AttributeError:
            print(
                'Internal method make_catalogue() must be run before '
                'running make_deltag_map().'
            )
        # Mean galaxy density
        maskbin = self.mask > 0
        nbar = nmap[maskbin].sum() / self.mask[maskbin].sum()
        # Map of expected galaxies per pixel
        nbar_map = nbar * self.mask
        # Shot noise
        self.shotnoise = 4. * np.pi * np.mean(self.mask) / (self.npix * nbar)
        # Overdensity map
        dg = np.zeros(self.npix)
        dg[maskbin] = nmap[maskbin] / nbar_map[maskbin] - 1.
        self.deltag_map = dg

    def weight_data_with_templates(self):
        '''
        Uses the templates associated with this simulation to assign weights
        to each source in the simulated catalogue.
        '''
        if self.templates is None:
            raise ValueError(
                'Templates must be provided upon initialisation in order'
                'to weight the data.'
            )
        # Calculate ratio of galaxy counts to the mean across footprint
        maskbin = self.mask > 0
        nmap = np.bincount(self.ipix_data, minlength=self.npix).astype(float)
        nbar = nmap[maskbin].sum() / self.mask[maskbin].sum()
        nbar_map = nbar * self.mask
        DELTA = np.zeros(self.npix)
        DELTA[maskbin] = nmap[maskbin] / nbar_map[maskbin]
        # Construct array containing systematics and an identity vector
        S = np.array(
            [np.ones(self.npix)] +
            [s[0] for s in self.templates]
        ).T
        # Construct matrix of dot products of pairs of templates
        Ns = len(self.templates) + 1
        M = np.zeros((Ns, Ns))
        for i in range(Ns):
            for j in range(Ns):
                M[i][j] = np.sum(S[:, j] * (self.mask ** 2) * S[:, i])
        # Invert the matrix
        iM = np.linalg.inv(M)
        # Construct a vector of dot products of data and templates
        V = np.zeros(Ns)
        for i in range(Ns):
            V[i] = np.sum(S[:, i] * (self.mask ** 2) * DELTA)
        # Construct weights map
        FS = S @ iM @ V
        W = 1 / FS
        # Evaluate W at position of each data point
        ipix = hp.ang2pix(self.nside, *self.pos_data, lonlat=True)
        self.w_data = W[ipix]

    def make_nmtfield_cat(self, deproject=False, compute_nlb=False):
        '''
        Makes an NmtFieldCatalogClustering object from the class properties,
        applying linear template deprojection if requested.

        Parameters
        ----------
        deproject: bool
            Whether or not to apply linear template deprojection.

        compute_nlb: bool
            Whether or not to compute the noise deprojection bias. Will do
            nothing regardless of value if deproject is False.
        '''
        temps = None
        # Ensure mask is not used if randoms are provided
        if self.pos_ran is not None:
            mask_to_use = None
            if deproject:
                if self.templates_ran is None:
                    self.templates_ran = np.array(
                        [t[0][self.ipix_ran] for t in self.templates]
                    )
                temps = self.templates_ran
        else:
            mask_to_use = self.mask
            if deproject:
                temps = self.templates.reshape((self.ntemp, self.npix))

        df = nmt.NmtFieldCatalogClustering(
            self.pos_data,
            self.w_data,
            positions_rand=self.pos_ran,
            weights_rand=self.w_ran,
            lmax=self.lmax,
            mask=mask_to_use,
            templates=temps,
            masked_on_input=self.is_masked,
            lonlat=True,
            calculate_noise_dp_bias=compute_nlb,
            lmax_deproj=self.lmax_deproj
        )
        self.field = df

    def make_nmtfield_map(self, deproject=False):
        '''
        Makes a map-based NmtField object from the simulated catalogue,
        via the creation of a density contrast map if necessary. Applies
        linear template deprojection if requested.

        Parameters
        ----------
        deproject: bool
            Whether or not to apply linear template deprojection.
        '''
        if deproject:
            temps = self.templates
        else:
            temps = None

        try:
            dg = self.deltag_map
        except AttributeError:
            self.make_deltag_map()

        df = nmt.NmtField(
                self.mask,
                [dg],
                templates=temps,
                masked_on_input=self.is_masked
            )
        self.field = df

    @classmethod
    def run_analysis_randoms(
        cls,
        cl,
        nside,
        ndata,
        mask,
        pos_ran,
        templates,
        alphas,
        templates_ran=None,
        wsp=None,
        nmt_alpha0=None,
        nmtbin=None,
        cls_to_compute=['all'],
        compute_nlb=False,
        lmax_deproj=None,
        seed=None
    ):
        '''
        Method for generating a Poisson realisations of a catalogue and
        computing angular power spectra for four scenarios:
            1. Cat is not contaminated, no deprojection applied (ncnd)
            2. Cat is not contaminated, but deprojection applied (ncd)
            3. Cat is contaminated, but no deprojection applied (cnd)
            4. Cat is contaminated, deprojection applied (cd).

        Parameters
        ----------
        cl: array[float]
            Input angular power spectrum, assumed to be defined at every
            multipole from 0 to (at least) 3 * nside - 1.

        nside: int
            HealPIX Nside value; determines resolution of the map.

        ndata: int
            The expectation value of the Poisson distribution from which the
            number of simulated sources will be drawn.

        mask: array[float]
            Map defining the survey geometry. Assumed to have the same
            resolution and format as the map to be created.

        pos_ran: array[float] or None
            Array containing RAs and Decs for a set of randoms.

        templates: array[float]
            Array containing any systematics templates with which the map is
            to be contaminated. Must have shape of either (Nsyst, 1, Npix) or
            (Nsyst, Npix), where Nsyst is the number of templates and Npix is
            the number of pixels in a map with the desired Nside.

        alphas: array[float]
            Contamination amplitudes for each template, i.e. for a given
            template t with amplitude alpha, the map will be contaminated
            by adding alpha*t.

        wsp: NmtWorkspace or None
            Workspace with stored mode-coupling matrix for use in decoupling
            computed pseudo-C_ells. If not provided, will construct one in
            situ.

        nmt_alpha0: float or None
            NmtFieldCatalogClustering alpha parameter, associated with whatever
            field was used to create the workspace provided as input (if any).

        nmtbin: NmtBin or None
            Object defining the bandpowers to use for decoupled C_ells.
            Ignored if wsp is not None. Cannot be None if wsp is None.

        cls_to_compute: list
            List of strings corresponding to the C_ells one wishes to compute.
            These strings must be in {'ncnd', 'ncd', 'cnd', 'cd', 'all'}. If
            'all', will compute all of the above options.

        compute_nlb: bool
            Whether or not to compute the noise deprojection bias.

        lmax_deproj: int or None
            Maximum multipole to which contaminants will be deprojected. If
            None, will default to the same lmax as is used for computing the
            C_ells generally (3 * N_side - 1).

        seed: int or None
            Seed to use when generating the map and catalogue.
        '''
        # Determine which C_ells to compute
        if 'all' in cls_to_compute:
            cls_to_compute = ['ncnd', 'ncd', 'cnd', 'cd']
        # Initialise simulation with non-contaminated map
        psim = cls(
            cl,
            nside,
            ndata,
            mask=mask,
            templates=templates,
            pos_ran=pos_ran,
            templates_ran=templates_ran,
            lmax_deproj=lmax_deproj,
            seed=seed
        )
        psim.make_catalogue()
        # Set up dictionary to store results
        psim.analysis = {}

        # No contamination, no deprojection (ncnd)
        if 'ncnd' in cls_to_compute:
            psim.make_nmtfield_cat(deproject=False, compute_nlb=False)
            # Construct NmtWorkspace if none provided
            if wsp is None:
                wsp = nmt.NmtWorkspace(psim.field, psim.field, nmtbin)
                nmt_alpha0 = psim.field.alpha
            psim.analysis['pcl_ncnd'] = nmt.compute_coupled_cell(
                psim.field, psim.field
            )
            mcm_renorm = psim.field.alpha / nmt_alpha0
            wsp.update_amplitude(mcm_renorm, mcm_renorm)
            psim.analysis['cl_ncnd'] = wsp.decouple_cell(
                psim.analysis['pcl_ncnd']
            )
            wsp.update_amplitude(1/mcm_renorm, 1/mcm_renorm)

        # No contamination, deprojection (ncd)
        if 'ncd' in cls_to_compute:
            psim.make_nmtfield_cat(deproject=True, compute_nlb=compute_nlb)
            if wsp is None:
                wsp = nmt.NmtWorkspace(psim.field, psim.field, nmtbin)
                nmt_alpha0 = psim.field.alpha
            psim.analysis['pcl_ncd'] = nmt.compute_coupled_cell(
                psim.field, psim.field
            )
            mcm_renorm = psim.field.alpha / nmt_alpha0
            wsp.update_amplitude(mcm_renorm, mcm_renorm)
            psim.analysis['cl_ncd'] = wsp.decouple_cell(
                psim.analysis['pcl_ncd']
            )
            if compute_nlb:
                psim.analysis['nlb_ncd'] = wsp.decouple_cell(
                    psim.field.clb
                )
            wsp.update_amplitude(1/mcm_renorm, 1/mcm_renorm)

        if not any(['cnd' in cls_to_compute, 'cd' in cls_to_compute]):
            return psim

        psim.contaminate_map(alphas=alphas)
        psim.make_catalogue()

        # Contamination, no deprojection (cnd)
        if 'cnd' in cls_to_compute:
            psim.make_nmtfield_cat(deproject=False, compute_nlb=False)
            if wsp is None:
                wsp = nmt.NmtWorkspace(psim.field, psim.field, nmtbin)
                nmt_alpha0 = psim.field.alpha
            psim.analysis['pcl_cnd'] = nmt.compute_coupled_cell(
                psim.field, psim.field
            )
            mcm_renorm = psim.field.alpha / nmt_alpha0
            wsp.update_amplitude(mcm_renorm, mcm_renorm)
            psim.analysis['cl_cnd'] = wsp.decouple_cell(
                psim.analysis['pcl_cnd']
            )
            wsp.update_amplitude(1/mcm_renorm, 1/mcm_renorm)

        # Contamination, deprojection (cd)
        if 'cd' in cls_to_compute:
            psim.make_nmtfield_cat(deproject=True, compute_nlb=compute_nlb)
            if wsp is None:
                wsp = nmt.NmtWorkspace(psim.field, psim.field, nmtbin)
                nmt_alpha0 = psim.field.alpha
            psim.analysis['pcl_cd'] = nmt.compute_coupled_cell(
                psim.field, psim.field
            )
            mcm_renorm = psim.field.alpha / nmt_alpha0
            wsp.update_amplitude(mcm_renorm, mcm_renorm)
            psim.analysis['cl_cd'] = wsp.decouple_cell(
                psim.analysis['pcl_cd']
            )
            if compute_nlb:
                psim.analysis['nlb_cd'] = wsp.decouple_cell(
                    psim.field.clb
                )
            wsp.update_amplitude(1/mcm_renorm, 1/mcm_renorm)
            psim.analysis['alphas'] = psim.field.alphas

        return psim

    @classmethod
    def run_analysis_mask(
        cls,
        cl,
        nside,
        ndata,
        mask,
        templates,
        alphas,
        wsp=None,
        nmt_alpha0=None,
        nmtbin=None,
        cls_to_compute=['all'],
        compute_nlb=False,
        lmax_deproj=None,
        seed=None
    ):
        '''
        Method for generating a Poisson realisations of a catalogue and
        computing angular power spectra for four scenarios:
            1. Cat is not contaminated, no deprojection applied (ncnd)
            2. Cat is not contaminated, but deprojection applied (ncd)
            3. Cat is contaminated, but no deprojection applied (cnd)
            4. Cat is contaminated, deprojection applied (cd).

        Parameters
        ----------
        cl: array[float]
            Input angular power spectrum, assumed to be defined at every
            multipole from 0 to (at least) 3 * nside - 1.

        nside: int
            HealPIX Nside value; determines resolution of the map.

        ndata: int
            The expectation value of the Poisson distribution from which the
            number of simulated sources will be drawn.

        mask: array[float]
            Map defining the survey geometry. Assumed to have the same
            resolution and format as the map to be created.

        templates: array[float]
            Array containing any systematics templates with which the map is
            to be contaminated. Must have shape of either (Nsyst, 1, Npix) or
            (Nsyst, Npix), where Nsyst is the number of templates and Npix is
            the number of pixels in a map with the desired Nside.

        alphas: array[float]
            Contamination amplitudes for each template, i.e. for a given
            template t with amplitude alpha, the map will be contaminated
            by adding alpha*t.

        wsp: NmtWorkspace or None
            Workspace with stored mode-coupling matrix for use in decoupling
            computed pseudo-C_ells. If not provided, will construct one in
            situ.

        nmt_alpha0: float or None
            NmtFieldCatalogClustering alpha parameter, associated with whatever
            field was used to create the workspace provided as input (if any).

        nmtbin: NmtBin or None
            Object defining the bandpowers to use for decoupled C_ells.
            Ignored if wsp is not None. Cannot be None if wsp is None.

        cls_to_compute: list
            List of strings corresponding to the C_ells one wishes to compute.
            These strings must be in {'ncnd', 'ncd', 'cnd', 'cd', 'all'}. If
            'all', will compute all of the above options.

        compute_nlb: bool
            Whether or not to compute the noise deprojection bias.

        lmax_deproj: int or None
            Maximum multipole to which contaminants will be deprojected. If
            None, will default to the same lmax as is used for computing the
            C_ells generally (3 * N_side - 1).

        seed: int or None
            Seed to use when generating the map and catalogue.
        '''
        # Determine which C_ells to compute
        if 'all' in cls_to_compute:
            cls_to_compute = ['ncnd', 'ncd', 'cnd', 'cd']
        # Initialise simulation with non-contaminated map
        psim = cls(
            cl,
            nside,
            ndata,
            mask=mask,
            templates=templates,
            lmax_deproj=lmax_deproj,
            seed=seed
        )
        psim.make_catalogue()
        # Set up dictionary to store results
        psim.analysis = {}

        # No contamination, no deprojection (ncnd)
        if 'ncnd' in cls_to_compute:
            psim.make_nmtfield_cat(deproject=False, compute_nlb=False)
            if wsp is None:
                wsp = nmt.NmtWorkspace(psim.field, psim.field, nmtbin)
                nmt_alpha0 = psim.field.alpha
            psim.analysis['pcl_ncnd'] = nmt.compute_coupled_cell(
                psim.field, psim.field
            )
            mcm_renorm = psim.field.alpha / nmt_alpha0
            wsp.update_amplitude(mcm_renorm, mcm_renorm)
            psim.analysis['cl_ncnd'] = wsp.decouple_cell(
                psim.analysis['pcl_ncnd']
            )
            wsp.update_amplitude(1/mcm_renorm, 1/mcm_renorm)

        # No contamination, deprojection (ncd)
        if 'ncd' in cls_to_compute:
            psim.make_nmtfield_cat(deproject=True, compute_nlb=compute_nlb)
            if wsp is None:
                wsp = nmt.NmtWorkspace(psim.field, psim.field, nmtbin)
                nmt_alpha0 = psim.field.alpha
            psim.analysis['pcl_ncd'] = nmt.compute_coupled_cell(
                psim.field, psim.field
            )
            mcm_renorm = psim.field.alpha / nmt_alpha0
            wsp.update_amplitude(mcm_renorm, mcm_renorm)
            psim.analysis['cl_ncd'] = wsp.decouple_cell(
                psim.analysis['pcl_ncd']
            )
            if compute_nlb:
                psim.analysis['nlb_ncd'] = wsp.decouple_cell(
                    psim.field.clb
                )
            wsp.update_amplitude(1/mcm_renorm, 1/mcm_renorm)

        if not any(['cnd' in cls_to_compute, 'cd' in cls_to_compute]):
            return psim

        psim.contaminate_map(alphas=alphas)
        psim.make_catalogue()

        # Contamination, no deprojection (cnd)
        if 'cnd' in cls_to_compute:
            psim.make_nmtfield_cat(deproject=False, compute_nlb=False)
            if wsp is None:
                wsp = nmt.NmtWorkspace(psim.field, psim.field, nmtbin)
                nmt_alpha0 = psim.field.alpha
            psim.analysis['pcl_cnd'] = nmt.compute_coupled_cell(
                psim.field, psim.field
            )
            mcm_renorm = psim.field.alpha / nmt_alpha0
            wsp.update_amplitude(mcm_renorm, mcm_renorm)
            psim.analysis['cl_cnd'] = wsp.decouple_cell(
                psim.analysis['pcl_cnd']
            )
            wsp.update_amplitude(1/mcm_renorm, 1/mcm_renorm)

        # Contamination, deprojection (cd)
        if 'cd' in cls_to_compute:
            psim.make_nmtfield_cat(deproject=True, compute_nlb=compute_nlb)
            if wsp is None:
                wsp = nmt.NmtWorkspace(psim.field, psim.field, nmtbin)
                nmt_alpha0 = psim.field.alpha
            psim.analysis['pcl_cd'] = nmt.compute_coupled_cell(
                psim.field, psim.field
            )
            mcm_renorm = psim.field.alpha / nmt_alpha0
            wsp.update_amplitude(mcm_renorm, mcm_renorm)
            psim.analysis['cl_cd'] = wsp.decouple_cell(
                psim.analysis['pcl_cd']
            )
            if compute_nlb:
                psim.analysis['nlb_cd'] = wsp.decouple_cell(
                    psim.field.clb
                )
            wsp.update_amplitude(1/mcm_renorm, 1/mcm_renorm)
            psim.analysis['alphas'] = psim.field.alphas

        return psim

    @classmethod
    def run_analysis_weights(
        cls,
        cl,
        nside,
        ndata,
        mask,
        pos_ran,
        templates,
        alphas,
        wsp=None,
        nmt_alpha0=None,
        nmtbin=None,
        cls_to_compute=['all'],
        seed=None
    ):
        '''
        Method for generating a Poisson realisations of a catalogue and
        computing angular power spectra for four scenarios:
            1. Cat is not contaminated, no deprojection applied (ncnd)
            2. Cat is not contaminated, but deprojection applied (ncd)
            3. Cat is contaminated, but no deprojection applied (cnd)
            4. Cat is contaminated, deprojection applied (cd).

        Parameters
        ----------
        cl: array[float]
            Input angular power spectrum, assumed to be defined at every
            multipole from 0 to (at least) 3 * nside - 1.

        nside: int
            HealPIX Nside value; determines resolution of the map.

        ndata: int
            The expectation value of the Poisson distribution from which the
            number of simulated sources will be drawn.

        mask: array[float]
            Map defining the survey geometry. Assumed to have the same
            resolution and format as the map to be created.

        pos_ran: array[float] or None
            Array containing RAs and Decs for a set of randoms.

        templates: array[float]
            Array containing any systematics templates with which the map is
            to be contaminated. Must have shape of either (Nsyst, 1, Npix) or
            (Nsyst, Npix), where Nsyst is the number of templates and Npix is
            the number of pixels in a map with the desired Nside.

        alphas: array[float]
            Contamination amplitudes for each template, i.e. for a given
            template t with amplitude alpha, the map will be contaminated
            by adding alpha*t.

        wsp: NmtWorkspace or None
            Workspace with stored mode-coupling matrix for use in decoupling
            computed pseudo-C_ells. If not provided, will construct one in
            situ.

        nmt_alpha0: float or None
            NmtFieldCatalogClustering alpha parameter, associated with whatever
            field was used to create the workspace provided as input (if any).

        nmtbin: NmtBin or None
            Object defining the bandpowers to use for decoupled C_ells.
            Ignored if wsp is not None. Cannot be None if wsp is None.

        cls_to_compute: list
            List of strings corresponding to the C_ells one wishes to compute.
            These strings must be in {'ncnd', 'ncd', 'cnd', 'cd', 'all'}. If
            'all', will compute all of the above options.

        seed: int or None
            Seed to use when generating the map and catalogue.
        '''
        # Determine which C_ells to compute
        if 'all' in cls_to_compute:
            cls_to_compute = ['ncnd', 'ncd', 'cnd', 'cd']
        # Initialise simulation with non-contaminated map
        psim = cls(
            cl,
            nside,
            ndata,
            mask=mask,
            templates=templates,
            pos_ran=pos_ran
        )
        psim.make_catalogue()
        # Set up dictionary to store results
        psim.analysis = {}

        # No contamination, no deprojection (ncnd)
        if 'ncnd' in cls_to_compute:
            psim.make_nmtfield_cat(deproject=False, compute_nlb=False)
            # Construct NmtWorkspace if none provided
            if wsp is None:
                wsp = nmt.NmtWorkspace(psim.field, psim.field, nmtbin)
                nmt_alpha0 = psim.field.alpha
            psim.analysis['pcl_ncnd'] = nmt.compute_coupled_cell(
                psim.field, psim.field
            )
            mcm_renorm = psim.field.alpha / nmt_alpha0
            wsp.update_amplitude(mcm_renorm, mcm_renorm)
            psim.analysis['cl_ncnd'] = wsp.decouple_cell(
                psim.analysis['pcl_ncnd']
            )
            wsp.update_amplitude(1/mcm_renorm, 1/mcm_renorm)

        # No contamination, deprojection (ncd)
        if 'ncd' in cls_to_compute:
            psim.weight_data_with_templates()
            psim.make_nmtfield_cat(deproject=False, compute_nlb=False)
            if wsp is None:
                wsp = nmt.NmtWorkspace(psim.field, psim.field, nmtbin)
                nmt_alpha0 = psim.field.alpha
            psim.analysis['pcl_ncd'] = nmt.compute_coupled_cell(
                psim.field, psim.field
            )
            mcm_renorm = psim.field.alpha / nmt_alpha0
            wsp.update_amplitude(mcm_renorm, mcm_renorm)
            psim.analysis['cl_ncd'] = wsp.decouple_cell(
                psim.analysis['pcl_ncd']
            )
            wsp.update_amplitude(1/mcm_renorm, 1/mcm_renorm)

        psim.contaminate_map(alphas=alphas)
        psim.make_catalogue()

        # Contamination, no deprojection (cnd)
        if 'cnd' in cls_to_compute:
            psim.make_nmtfield_cat(deproject=False, compute_nlb=False)
            if wsp is None:
                wsp = nmt.NmtWorkspace(psim.field, psim.field, nmtbin)
                nmt_alpha0 = psim.field.alpha
            psim.analysis['pcl_cnd'] = nmt.compute_coupled_cell(
                psim.field, psim.field
            )
            mcm_renorm = psim.field.alpha / nmt_alpha0
            wsp.update_amplitude(mcm_renorm, mcm_renorm)
            psim.analysis['cl_cnd'] = wsp.decouple_cell(
                psim.analysis['pcl_cnd']
            )
            wsp.update_amplitude(1/mcm_renorm, 1/mcm_renorm)

        # Contamination, deprojection (cd)
        if 'cd' in cls_to_compute:
            if wsp is None:
                wsp = nmt.NmtWorkspace(psim.field, psim.field, nmtbin)
                nmt_alpha0 = psim.field.alpha
            psim.weight_data_with_templates()
            psim.make_nmtfield_cat(deproject=True, compute_nlb=False)
            psim.analysis['pcl_cd'] = nmt.compute_coupled_cell(
                psim.field, psim.field
            )
            mcm_renorm = psim.field.alpha / nmt_alpha0
            wsp.update_amplitude(mcm_renorm, mcm_renorm)
            psim.analysis['cl_cd'] = wsp.decouple_cell(
                psim.analysis['pcl_cd']
            )
            wsp.update_amplitude(1/mcm_renorm, 1/mcm_renorm)

        return psim

    @classmethod
    def run_analysis_map(
        cls,
        cl,
        nside,
        ndata,
        mask,
        templates,
        alphas,
        wsp=None,
        nmtbin=None,
        cls_to_compute=['all'],
        compute_db_true=False,
        compute_db_guess=False,
        seed=None
    ):
        '''
        Method for generating a Gaussian sim and computing angular power
        spectra for four scenarios:
            1. Map is not contaminated, no deprojection applied (ncnd)
            2. Map is not contaminated, but deprojection applied (ncd)
            3. Map is contaminated, but no deprojection applied (cnd)
            4. Map is contaminated, deprojection applied (cd).

        Parameters
        ----------
        cl: array[float]
            Input angular power spectrum, assumed to be defined at every
            multipole from 0 to (at least) 3 * nside - 1.

        nside: int
            HealPIX Nside value; determines resolution of the map.

        mask: array[float]
            Map defining the survey geometry. Assumed to have the same
            resolution and format as the map to be created.

        templates: array[float]
            Array containing any systematics templates with which the map is
            to be contaminated. Must have shape of either (Nsyst, 1, Npix) or
            (Nsyst, Npix), where Nsyst is the number of templates and Npix is
            the number of pixels in a map with the desired Nside.

        alphas: array[float]
            Contamination amplitudes for each template, i.e. for a given
            template t with amplitude alpha, the map will be contaminated
            by adding alpha*t.

        wsp: NmtWorkspace or None
            Workspace with stored mode-coupling matrix for use in decoupling
            computed pseudo-C_ells. If not provided, will construct one in
            situ.

        nmtbin: NmtBin or None
            Object defining the bandpowers to use for decoupled C_ells.
            Ignored if wsp is not None. Cannot be None if wsp is None.

        cls_to_compute: list
            List of strings corresponding to the C_ells one wishes to compute.
            These strings must be in {'ncnd', 'ncd', 'cnd', 'cd', 'all'}. If
            'all', will compute all of the above options.

        compute_db_true: bool
            Whether or not to compute deprojection bias using the input C_ell
            as an estimate of the true C_ell.

        compute_db_guess: bool
            Whether or not to compute deprojection bias using the measured
            C_ell divided by the sky fraction as an estimate of the true C_ell.

        seed: int or None
            Seed to use when generating the map and catalogue.
        '''
        # Determine which C_ells to compute
        if 'all' in cls_to_compute:
            cls_to_compute = ['ncnd', 'ncd', 'cnd', 'cd']
        # Initialise simulation with non-contaminated map
        psim = cls(
            cl,
            nside,
            ndata,
            mask=mask,
            templates=templates,
            seed=seed
        )
        psim.make_catalogue()
        psim.make_deltag_map()
        # Set up dictionary to store results
        psim.analysis = {}

        # No contamination, no deprojection (ncnd)
        if 'ncnd' in cls_to_compute:
            psim.make_nmtfield_map(deproject=False)
            # Construct NmtWorkspace if none provided
            if wsp is None:
                wsp = nmt.NmtWorkspace.from_fields(
                    psim.field, psim.field, nmtbin
                )
            psim.analysis['pcl_ncnd'] = nmt.compute_coupled_cell(
                psim.field, psim.field
            )
            psim.analysis['cl_ncnd'] = wsp.decouple_cell(
                psim.analysis['pcl_ncnd']
            )
            psim.analysis['pnl_nc']\
                = np.ones((1, psim.lmax+1)) * psim.shotnoise
            psim.analysis['nl_nc'] = wsp.decouple_cell(
                psim.analysis['pnl_nc']
            )

        # No contamination, deprojection (ncd)
        if 'ncd' in cls_to_compute:
            psim.make_nmtfield_map(deproject=True)
            if wsp is None:
                wsp = nmt.NmtWorkspace.from_fields(
                    psim.field, psim.field, nmtbin
                )
            psim.analysis['pcl_ncd'] = nmt.compute_coupled_cell(
                psim.field, psim.field
            )
            psim.analysis['cl_ncd'] = wsp.decouple_cell(
                psim.analysis['pcl_ncd']
            )

        if not any(['cnd' in cls_to_compute, 'cd' in cls_to_compute]):
            return psim

        psim.contaminate_map(alphas=alphas)
        psim.make_catalogue()
        psim.make_deltag_map()

        # Contamination, no deprojection (cnd)
        if 'cnd' in cls_to_compute:
            psim.make_nmtfield_map(deproject=False)
            if wsp is None:
                wsp = nmt.NmtWorkspace.from_fields(
                    psim.field, psim.field, nmtbin
                )
            psim.analysis['pcl_cnd'] = nmt.compute_coupled_cell(
                psim.field, psim.field
            )
            psim.analysis['cl_cnd'] = wsp.decouple_cell(
                psim.analysis['pcl_cnd']
            )
            psim.analysis['pnl_c'] = np.ones((1, psim.lmax+1)) * psim.shotnoise
            psim.analysis['nl_c'] = wsp.decouple_cell(
                psim.analysis['pnl_c']
            )

        # Contamination, deprojection (cd)
        if 'cd' in cls_to_compute:
            psim.make_nmtfield_map(deproject=True)
            if wsp is None:
                wsp = nmt.NmtWorkspace.from_fields(
                    psim.field, psim.field, nmtbin
                )
            psim.analysis['pcl_cd'] = nmt.compute_coupled_cell(
                psim.field, psim.field
            )
            psim.analysis['cl_cd'] = wsp.decouple_cell(
                psim.analysis['pcl_cd']
            )
            # Deprojection bias
            if compute_db_true:
                psim.analysis['clb_true'] = wsp.decouple_cell(
                    nmt.deprojection_bias(
                        psim.field,
                        psim.field,
                        cl.reshape(1, -1)
                    )
                )
            if compute_db_guess:
                psim.analysis['clb_guess'] = wsp.decouple_cell(
                    nmt.deprojection_bias(
                        psim.field,
                        psim.field,
                        psim.analysis['pcl_cd'] / psim.fsky
                    )
                )
            psim.analysis['alphas'] = psim.field.alphas

        return psim
