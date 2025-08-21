#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 12:29:21 2022
fidA_processing.py

@author: cbailey, based on Matlab code by Jamie Near
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift,ifft
from scipy.optimize import curve_fit
GAMMA_DICT={'1H':42.577,'2H':6.536,'13C':10.7084,'19F':40.078,'23Na':11.262,'31P':17.235}
import functools

def add_phase(invec,added_phase):
    """
    Add equal amounts of complex phase to each point of a vector (note this 
    function operates on a numpy array rather than a fid object. To operate on 
    the fid object, use 'op_addphase').

    Parameters
    ----------
    invec : Input numpy array
    added_phase : float
        Amount of phase (in degrees) to add.

    Returns
    -------
    output vector
        0th order phased version of the input.

    """
    return invec*np.exp(1j*added_phase*np.pi/180)

def add_phase1(invec,ppm,timeShift,ppm0=4.65,B0=7,nucleus='1H'):
    """
    Add first order phase to a spectrum (added phase is linearly dependent on 
    frequency). This function operates on a numpy array, not a fid-A object. For a
    phase shifting function that operates on a fid-A object, see 'op_addphase'

    Parameters
    ----------
    invec : Input numpy array (spectrum)
    ppm : 1D vector
        Frequency scale (ppm). The length of ppm should match the first dimension
        of invec so that multiplication with invec.T can be done via broadcasting
        (https://numpy.org/doc/stable/user/basics.broadcasting.html) and then the
        transposed result returned. If ppm has unnecessary singleton dimensions,
        eg. [n,1] while invec is [n,3], then multiplication will fail
    timeShift : float
        Amount of 1st order phase shift (specified as horizontal shift in 
        seconds in the time domain).
    ppm0 : float, optional
        The frequency "origin" (in ppm) of the 1st order phase shift (this
        point will undergo 0 phase shift). The default is 4.65.
    B0 : float, optional
        Magnetic field strength in Tesla (needed to convert ppm to Hz). The default is 7.
    nucleus : string, optional
        The nucleus that will be used to determine the gyromagnetic ratio for
        the ppm to Hz conversion. The 'nucleus' string is used as a key for
        GAMMA_DICT. The default is '1H'.

    Returns
    -------
    phased_spec : Output numpy array
        1st order phased version of the input.

    """
    f=(ppm-ppm0)*GAMMA_DICT[nucleus.upper()]*B0
    # f is in Hz and timeshift in s, so multiplyling them gives result in 
    # cycles. Multiply by 2*pi to get phase in radians
    phased_spec=(invec.T*np.exp(-1j*f*timeShift*2*np.pi)).T
    return phased_spec

def op_addNoise(indat,sdnoise):
    """
    Add noise to a spectrum. Useful for simulated data

    Parameters
    ----------
    indat : FID object
        input data.
    sdnoise : float or numpy array
        If float, the standard deviation of Gaussian noise to be added. If numpy
        array, the (complex) values to be added onto indat.fids.

    Returns
    -------
    outdat : FID class.
        output data with noise added
    noisevec : numpy array
        The array of Gaussian noise values that were added to the data
    """
    outdat=indat.copy()
    if type(sdnoise) is np.ndarray:
        noisevec=sdnoise
    else:
        noisevec=sdnoise*np.random.randn(indat.sz)+1j*sdnoise*np.random.randn(indat.sz)
    outdat.fids=indat.fids+noisevec
    return outdat,noisevec

def op_addphase(indat,ph0,ph1=0,ppm0=4.65,suppress_plot=True):
    """
    Add 0th and/or 1st order phase to a the spectrum of a FID object.

    Parameters
    ----------
    indat : FID object
        Phase will be added to this spectrum.
    ph0 : float
        Zeroth-order phase (in degrees) to be added to FID.fids.
    ph1 : float, optional
        First-order phase to be added onto FID.specs. The default is 0.
    ppm0 : float, optional
        The center point at which to calculate the first order phase. The default is 4.65.
    suppress_plot : boolean, optional
        Whether to suppress the plot of the final spectrum. Only spectra with
        fewer than 3 dimensions will be plotted. The default is True.

    Returns
    -------
    outdat : FID object
        A new FID object with the ph0 and ph1 corrections.

    """
    outdat=indat.copy()
    outdat.fids=indat.fids*np.exp(1j*ph0*np.pi/180)
    outdat.added_ph0=indat.added_ph0+ph0
    #Now add 1st-order phase
    outdat.specs=add_phase1(outdat.specs,indat.ppm,ph1,ppm0,indat.Bo,indat.nucleus[0])
    outdat.added_ph1=indat.added_ph1+ph1
    if outdat.ndim<3 and not suppress_plot:
        outdat.plot_spec(xlims=[outdat.ppm[0],outdat.ppm[-1]])
    return outdat

def op_addphaseSubspec(indat,ph0):
    """
    For spectra with two subspectra, add zero order phase to the second 
    subspectra in a dataset. With edited spectroscopy sequences (eg. mega-
    press), there can be small frequency drifts between edit-on and edit-off 
    spectra that can result in residual signals from uncoupled spins 
    (Cr, Ch, etc).

    Parameters
    ----------
    indat : FID object
        Input data with two subspectra.
    ph0 : float
        Phase (in degrees) to add to the second subspectrum.

    Returns
    -------
    outdat : FID object
        Output dataset with phase adjusted subspectrum..

    """
    if indat.dims['coils']>=0:
        raise TypeError('ERROR: Cannot operate on data with multilple coils! ABORTING!!')
    elif indat.dims['averages']>=0:
        raise TypeError('ERROR: Cannot operate on data with multiple averages! ABORTING!!')
    elif indat.dims['subSpecs']<0:
        raise TypeError('ERROR: Can not operate on data with no Subspecs! ABORTING!!')
    if indat.sz[indat.dims['subSpecs']]!=2:
        raise TypeError('ERROR: Input spectrum must have two subspecs! ABORTING!!')
    outdat=indat.copy()
    # This slice1 construction should allow the 2nd spectrum of the subSpec 
    # dimension to be selected for calculation. Keeps function generalizable.
    slice1=[slice(None)]*outdat.ndim
    slice1[outdat.dims['subSpecs']]=1
    outdat.fids[tuple(slice1)]=outdat.fids[tuple(slice1)]*np.exp(1j*ph0*np.pi/180)
    return outdat

def op_addrcvrs(indat,phasept=0,mode='w',coilcombos=None):
    """
    Perform weighted coil recombination for MRS data acquired with receiver
    coil array.

    Parameters
    ----------
    indat : FID object
        Input data with multiple receiver info.
    phasept : float, optional
        Point of fid to use for phase estimation and amplitude when mode='w'. 
        The default is 0.
    mode : char, optional
        Method for estiamting the coil weights and phases if not provided in 
        coilcombos. Can be:
            'w' - performs amplitude weighting of channels based on the max 
                signal of each coil channel
            'h' - performs amplitude weighting of channles based on the max signal 
                of each coil channel divided by the square of the noise in each 
                coil channel (as described by Hall et al. Neuroimage 2014).
            The default is 'w'.
    coilcombos : dict with keys 'phs' and 'sigs', optional
        The predetermined coil phases (in degrees) and amplitudes as generated
        by op_getcoilcombos. If this argument is provided, the 'point' and 'mode'
        arguments will be ignored. The default is None.

    Returns
    -------
    outdat : FID object
        The output dataset with coil channels combined.
    fids_presum : numpy array
        Input time domain data (fid) with coils phase-adjusted, before combination.
    specs_presum : numpy array
        Input frequency domain data (spectrum) with coils phase-adjusted, before 
        combination.
    coilcombos : dict with keys 'phs' and 'sigs'
        The vectors of the coil phases (in degrees) used for alignment and the 
        coil weights.

    """
    if indat.flags['addedrcvrs'] or indat.dims['coils']==-1 or indat.sz[indat.dims['coils']]==1:
        print('WARNING: Only one receiver channel found! Returning input without modification!')
        outdat=indat.copy()
        outdat.flags['addedrcvrs']=True
        fids_presum=indat.fids
        specs_presum=indat.specs
        coilcombos={'phs':0, 'sigs':1}
    else:
        #To get best possible SNR, add the averages together (if it hasn't already been done):
        if not indat.flags['averaged']:
            av=op_averaging(indat)
        else:
            av=indat.copy()
        # also, for best results, we will combine all subspectra:
        if coilcombos is None:
            if indat.flags['isFourSteps']:
                av=op_fourStepCombine(av)
            if indat.dims['subSpecs']>-1:
                av=op_combinesubspecs(av,'summ')
        if coilcombos is None:
            # Code was repeated in Matlab rather than calling function
            coilcombos=op_getcoilcombos(av,phasept=phasept,mode=mode)
        phs=coilcombos['phs']
        sigs=coilcombos['sigs']/np.linalg.norm(coilcombos['sigs'].flatten())
        # now expand these matrices to match the size of indat.fids for 
        # multiplication. Each coil has a different phase and amplitude
        ph=np.ones(indat.sz)
        sig=np.ones(indat.sz)
        slice1=[slice(None)]*indat.fids.ndim
        for nct in range(indat.sz[indat.dims['coils']]):
            slice1[indat.dims['coils']]=nct
            ph[tuple(slice1)]=phs[nct]
            sig[tuple(slice1)]=sigs[nct]
        
        # now apply the phases by multiplying the data by exp(-i*ph);
        fids=indat.fids*np.exp(-1j*ph*np.pi/180)
        fidobj_presum=indat.copy()
        fidobj_presum.fids=fids
        fids_presum=fidobj_presum.fids
        specs_presum=fidobj_presum.specs
        fids=fids*sig
        #Make the coilcombos structure:
        coilcombos={'phs':phs,'sigs':sigs}
        #now sum along coils dimension
        fids=np.sum(fids,axis=indat.dims['coils'])
        outdat=indat.copy()
        outdat.fids=np.squeeze(fids)
        # change the dims variables
        outdat._remove_dim_from_dict('coils')
        outdat.flags['addedrcvrs']=True
        outdat.flags['writtentostruct']=True      
    return outdat,fids_presum,specs_presum,coilcombos

def op_addScans(indat1,indat2,subtract=False):
    """
    Add or subtract two scans. This function exists for those replicating
    Matlab work. However, the __add__ method is defined for the FID object 
    so that users can just call indat1+indat2 or indat1-indat2 directly.

    Parameters
    ----------
    indat1 : FID object
        First spectrum to add.
    indat2 : FID object
        Second spectrum to add.
    subtract : BINARY, optional
        Indicates whether to or subtract (True or non-zero int) or add (False 
        or 0) the two spectra. The default is False.

    Returns
    -------
    outdat : FID object
        The output resulting from adding (or subtracting) indat1 and indat2.

    """
    # I don't fully understand this case but Jamie includes it in Matlab for the
    # case of looping through multiple spectra. However, seems to assume that
    # input will be an empty structure for index 0, whereas Python is itself
    # 0-based for arrays. So this may not be what I want. Any such loops in 
    # processing functions should likely just use Python's sum function with an
    # appropriate start value. eg. sum(list_of_fids,start=0*list_of_fids[0])
    if indat1 is None:
        indat1=0*indat2
    if indat1.sz != indat2.sz:
        raise Exception('ERROR:  Spectra must be the same number of points')
    if indat1.spectralwidth != indat2.spectralwidth:
        raise Exception('ERROR:  Spectra must have the same spectral width')
    # No need to check dwelltime since it is automatically defined by spectral 
    # width, but I've added a check to central frequency, which is the final 
    # check needed to ensure that both spectra have the same ppm.
    if indat1.center_freq_ppm != indat2.center_freq_ppm:
        raise Exception('ERROR:  Spectra must have the same central frequency')
    if subtract: #addition case
        outdat=indat1-indat2
    else: # subtraction case
        outdat=indat1+indat2
    return outdat

# Matlab fidA has a huge number of align functions. I have attempted to simplify
# and avoid duplicating code. However, I did create functions with the Matlab
# names for those familiar with the Matlab calls. The "fundamental" alignment
# function here in Python with the fitting functions is op_alignScans
def op_alignAllScans(inlist, tmax=0.5, ref='f', mode='fp',freq_range=None):
    # Make sure input is a list of length 2 or greater
    if type(inlist) is not list or len(inlist)<2:
        TypeError('ERROR: The input must be a list of two ore more MRS datasets in FID-A FID object form. ABORTING!!')
    # Figure out what reference spectrum will be
    if ref=='f':
        inref=inlist[0]
    elif ref=='a':
        inref=sum(inlist,inlist[0]*0)/len(inlist) # you have to provide a start value for sum of type FID. Otherwise it uses 0
    else:
        TypeError('ERROR: Reference spectrum not recognized.')
    ampref=np.max(np.abs(inref))
    outlist=[inlist[0]]
    phlist=[0]; frqlist=[0]
    for eachspec in inlist[1:]:
        amp1=np.max(np.abs(eachspec))
        [dummyOut,dummy_ph,dummy_frq]=op_alignScans(inref, eachspec*ampref/amp1, tmax=tmax, mode=mode, freq_range=freq_range)
        phlist.append(dummy_ph)
        frqlist.append(dummy_ph)
        outlist.append(op_addphase(op_freqshift(eachspec,dummy_frq),dummy_ph))
    return outlist, phlist, frqlist

def op_alignAllScans_fd(inlist, fmin, fmax, tmax=0.5, ref='f', mode='fp'):
    outlist,phlist,frqlist=op_alignAllScans(inlist, tmax, ref=ref, mode=mode,freq_range=[fmin,fmax])
    return outlist, phlist, frqlist
    
def op_alignAverages(indat,tmax=None,med='n',ref=None,freq_range=None):
    outdat=indat.copy()
    fs=0; phs=0;
    if not indat.flags['addedrcvrs']:
        print('ERROR: I think it only makes sense to do this after ou have combined the channels using op_addrcvrs. ABORTING!!')
    elif indat.dims['averages']==-1 or indat.sz[indat.dims['averages']]==1:
        print('WARNING: No averages found. Returning input without modification')
    else:
        if tmax is None:
            print('tmax not supplied. Calculating when SNR drops below 5...')
            sig=np.abs(indat.fids)
            noise=np.std(np.real(indat.fids[np.ceil(0.75*indat.shape[0]):,...]),axis=0)
            noise=np.mean(noise.flatten())
            tmaxest=np.zeros([sig.shape[0],])
            slice1=[0]*sig.ndim
            slice1[0]=slice(None)
            for rowct in sig.shape[0]:
                slice1[1]=rowct
                tmaxest[rowct]=indat.t[np.nonzero(sig[tuple(slice1)]/noise>5)[0][-1]]
            tmax=np.median(tmaxest)
            print('tmax = {:.2e} ms'.format(tmax*1000))
        if med.lower()=='r' and ref is None:
            raise Exception("ERROR:  If using the 'r' option for input variable 'med', then an argument for 'ref' must be provided")
        if indat.dims['subSpecs']==-1:
            B=1
        else:
            B=indat.sz[indat.dims['subSpecs']]
        # Note that this function is not as generalizable as some others. It 
        # assumes that you only have averages and/or subSpecs, with the averages
        # dimension before subSpecs. This makes some sense because the fitting
        # function needs a 1D vector to run least squares minimization so you 
        # won't have "extra" dimensions, although order isn't necessarily known.
        # For SVS, multi-receiver data are ruled out by the addedcrvrs check but
        # spatial data from MRS are not.
        fs=np.zeros([indat.sz[indat.dims['averages']],B]) # Will squeeze everything before return
        phs=np.zeros_like(fs)
        newfid=np.zeros([indat.sz[indat.dims['t']],indat.sz[indat.dims['averages']],B])
        for mct in range(B):
            # Create a slice to select the mct'th subspec
            subspec_pts=[slice(None)]*indat.ndim
            if indat.dims['subSpecs']!=-1:
                subspec_pts[indat.dims['subSpecs']]=mct
            # after this, tmpfid should no longer have a subSpecs dimension, even if indat did (removed during slicing)
            tmpfid=indat[tuple(subspec_pts)]
            if med.lower()=='y':
                ref2=op_median(tmpfid)
                indmin=-1
            elif med.lower()=='a':
                ref2=op_averaging(tmpfid)
                indmin=-1
            elif med.lower()=='n':
                # This function isn't finished. Also, Jamie seems to write it in
                # to op_alignAverages separately rather than calling the function.
                # So will need to check that what the function returns is what 
                # is needed by op_alignAverages
                ref2,metric,badavgs=op_rm_bad_averages(tmpfid)
                # Is this right or how does the shape of metric behave when tmpfid has subspecs vs doesn't?
                try:
                    indmin=np.argmin(metric[:,mct])
                except IndexError:
                    indmin=np.argmin(metric)
            elif med.lower()=='f':
                # Not sure why this option isn't available in Matlab but makes
                # sense to add it
                first_pts=[slice(None)]*tmpfid.ndim
                first_pts[tmpfid.dims['averages']]=0
                ref2=tmpfid[tuple(first_pts)]
                indmin=0
            elif med.lower()=='r':
                # The assumption here is that, if indat has subSpecs, then ref 
                # also has subSpecs and each subSpec is meant to be aligned to its
                # own subSpec reference
                ref2=ref[tuple(subspec_pts)]
                indmin=-1
            else:
                raise TypeError("ERROR: Invalid value for 'med'. Allowed values are 'y', 'a', 'n', 'f' and 'r'.")
            for avct in range(indat.sz[indat.dims['averages']]):
                # I've defined the size and dimension order of newfid already so
                # I can define this slicing object directly. But there may be problems
                # if indat.dims has subSpecs before averages.
                newfidslice=[slice(None)]*tmpfid.ndim
                newfidslice[tmpfid.dims['averages']]=avct
                #newfidslice=tuple([slice(None),avct,mct])
                if avct==indmin:
                    whichpts2=[slice(None)]*tmpfid.ndim
                    whichpts2[tmpfid.dims['averages']]=indmin
                    newfid[newfidslice]=tmpfid[tuple(whichpts2)]
                    fs[avct,mct]=0
                    phs[avct,mct]=0
                else:
                    tmpobj,phs[avct,mct],fs[avct,mct]=op_alignScans(ref2, tmpfid[tuple(newfidslice)], tmax=tmax, mode='fp', freq_range=freq_range)
                    newfid[:,avct,mct]=tmpobj.fids
        if outdat.dims['averages']>outdat.dims['subSpecs'] and outdat.dims['subSpecs']!=-1:
            # I don't think it should be the case for any fidA_io loaders, which 
            # should put averages before subSpecs, but just in case, this will
            # keep the original dimensions and change the order of newfid to match
            outdat.fids=np.transpose(newfid,[0,2,1])
        else:
            outdat.fids=np.squeeze(newfid)
        outdat.flags['freqcorrected']=True
        return outdat,np.squeeze(fs),np.squeeze(phs)
    
def op_alignAverages_fd(indat, minppm, maxppm, tmax=None, med='n', ref=None):
    out1,ph,frq=op_alignAverages(indat, tmax=tmax, med=med, ref=ref, freq_range=[minppm, maxppm])
    return out1, ph, frq

def op_alignISIS(indat,tmax=0.5, freq_range=None, initPars=None):
    # I don't think I can pass to alignScans here because I only phase the second
    # spectrum before combining
    def freqPhaseShiftComplexNest(in2,f,p):
        t=np.linspace(0,len(in2)*infloat_range.dwelltime,len(in2)+1)[:-1]
        shifted=add_phase(in2[:,1]*np.exp(-1j*t.T*f*2*np.pi),p)
        subtracted=(input[:,0]+shifted)/2
        y=np.r_[np.real(subtracted),np.imag(subtracted)]
        return y
    
    if indat.flags['addedrcvrs']:
        raise Exception('ERROR: I think it only makes sense to do this after you have combined the channels using op_addrcvrs. ABORTING!')
    if indat.dims['subSpecs']==-1:
        raise Exception('ERROR: Must have multiple subspectra. ABORTING!')
    # Okay, I think that I somewhat get this in that there should only ever be
    # a max dimension size of 2 in the subspec dimension because different reps
    # of subspec pairs will be contained in the 'averages' dimension. And so if
    # you want to align all subspecs then you basically go through an align each
    # pair of subspecs, then combine subspecs, then call op_alignAverages to align each
    # rep. BUT it still makes no sense to me that the base0 that we're aligning
    # to is the combined subspec and then we're only aligning the second subspec.
    # Surely, you would align to the paired first subspec OR to the average of 
    # all second subspecs or something. Here, if there is only one average, then
    # the least squares should always give you [0,0]?? Anyway, op_alignMPSubspecs
    # seems to work more like I would expect
    if indat.dims['averages']>-1:
        fs=np.zeros(indat.sz[indat.dims['averages']])
        phs=np.zeros_like(fs)
        #newsize=[eachdim for dimct,eachdim in enumerate(indat.sz) if dimct!=indat.dims['averages']]
    newfid=np.zeros(indat.sz)
    if initPars is None:
        initPars=[0]*2
    if tmax>indat.t[-1]:
        tmax=indat.t[-1]
    print('Aligning all averages to the Average ISIS subtracted spectrum')
    if indat.dims['averages']>-1:
        base0=op_median(op_combinesubspecs(indat, 'diff'))
    else:
        base0=op_combinesubspecs(indat, 'diff')
    if freq_range is not None:
        inref_range=op_freqrange(base0, freq_range[0], freq_range[1])
        infloat_range=op_freqrange(indat, freq_range[0], freq_range[1])
    else:
        inref_range=base0
        infloat_range=indat
    baseref=np.r_[np.real(inref_range.fids[np.logical_and(inref_range.t>=0,inref_range.t<tmax)]),np.imag(inref_range.fids[np.logical_and(inref_range.t>=0,inref_range.t<tmax)])]
        
    if indat.dims['averages']>-1:
        for avct in range(indat.dims['averages']):
            av_slice=[slice(None)]*infloat_range.ndim
            av_slice[indat.dims['averages']]=avct
            av_slice[indat.dims['t']]=slice(np.argwhere(inref_range.t>=0).squeeze()[0],np.argwhere(inref_range.t<tmax).squeeze()[-1])
            parsFit,pcov=curve_fit(freqPhaseShiftComplexNest,infloat_range.fids[tuple(av_slice)].squeeze(),baseref.squeeze(),initPars, maxfev=5000)
            fs[avct]=parsFit[0]; phs[avct]=parsFit[1];
            spec_slice=[slice(None)]*infloat_range.ndim
            spec_slice[indat.dims['averages']]=avct
            spec_slice[indat.dims['subSpecs']]=1
            newfid[tuple(spec_slice)]=add_phase(infloat_range.fids[tuple(spec_slice)]*np.exp(-1j*infloat_range.t*fs[avct]*2*np.pi),phs[avct])
            spec_slice[indat.dims['subSpecs']]=0
            newfid[tuple(spec_slice)]=infloat_range.fids[tuple(spec_slice)]
    else:
        parsFit,pcov=curve_fit(freqPhaseShiftComplexNest,infloat_range.fids[np.logical_and(infloat_range.t>=0,infloat_range.t<tmax),...].squeeze(),baseref.squeeze(),initPars, maxfev=5000)
        fs=parsFit[0]; phs=parsFit[1];
        spec_slice=[slice(None)]*infloat_range.ndim
        spec_slice[infloat_range.dims['subSpecs']]=1
        newfid[tuple(spec_slice)]=add_phase(infloat_range[tuple(spec_slice)].fids*np.exp(-1j*infloat_range.t*fs*2*np.pi),phs)
        spec_slice[infloat_range.dims['subSpecs']]=0
        newfid[tuple(spec_slice)]=infloat_range.fids[tuple(spec_slice)]
    outdat=indat.copy()
    outdat.fids=newfid
    outdat.flags['freqcorrected']=True
    return outdat, phs, fs

def op_alignMPSubpsecs(indat,mode='o',initPars=None,ppmWeights=None,freq_range=None):
    # No tmax since work is done on the frequency spectrum
    # This one also needs its own fitting function because it is comparing the
    # spectra, not the fid, and the weights are for the points on the spectrum
    def freqPhaseShiftComplexNest(in2,f,p):
        t=np.linspace(0,len(in2)*infloat.dwelltime,len(in2)+1)[:-1]
        shiftedFids=add_phase(in2*np.exp(-1j*t.T*f*2*np.pi),p)
        shiftedSpecs=fftshift(ifft(shiftedFids,axis=0),axes=0)
        y=np.r_[np.real(shiftedSpecs),np.imag(shiftedSpecs)]
        return y
    if not indat.flags['addedrcvrs']:
        raise Exception('ERROR: I think it only makes sense to do this after you have combined the channels using op_addrcvrs. ABORTING!!')
    if indat.dims['subSpecs']==-1:
        raise Exception('ERROR: Must have multiple subspectra. ABORTING!!')
    if indat.dims['averages']==-1:
        # I think you could loop averages like in op_alignISIS. Why is this not done?
        raise Exception('ERROR: Signal averaging must be performed before this step. ABORTING!!')
    if not (indat.sz==2 and indat.sz[indat.dims['subSpecs']]==2):
        raise Exception('ERROR: Expecting FID object with 2-dimensions and subSpecs dimension with size 2')
    if ppmWeights is None:
        ppmWeights=np.ones_like(indat.ppm)
    if initPars is None:
        initPars=[0]*2
    # Do basic error check on ppmWeights
    if len(ppmWeights)!=len(indat.ppm) or np.min(ppmWeights)<0:
        raise Exception('ERROR: ppmWeights must be a vector of real positive weights the same size as indat.ppm')
    if freq_range is not None:
        inpart=op_freqrange(indat, freq_range[0], freq_range[1])
    else:
        inpart=indat
    # Now adjust weights to be the same size as the frequency-limited spectrum
    if freq_range is not None:
        idxs=np.argwhere((indat.ppm>freq_range[0]) * (indat.ppm<freq_range[1])).squeeze()
        ppmWeights=ppmWeights[idxs]
    # Normalize weights
    ppmWeights=ppmWeights/np.sum(ppmWeights)
    print('Aligning the MEGA-PRESS edit-ON sub-spectrum to the edit-OFF sub-spectrum')
    base0=op_takesubspec(inpart,0)
    base=np.r_[np.real(base0.specs),np.imag(base0.specs)]
    infloat=op_takesubspec(inpart, 1)
    # Matlab's nlinfit uses weights, while scipy's curve_fit weights residuals based
    # on sigma=1/sqrt(w)
    parsFit,pcov=curve_fit(freqPhaseShiftComplexNest,infloat.fids,base,initPars, sigma=1/(ppmWeights), maxfev=5000)
    fs=parsFit[0]; phs=parsFit[1];
    if mode.lower()=='o':
        phs=phs+180
    newfid2=add_phase(infloat.fids*np.exp(-1j*infloat.t*fs*2*np.pi),phs)
    outdat=indat.copy()
    outdat.fids[:,1]=newfid2
    outdat.flags['freqcorrected']=True
    return outdat,fs,phs
    
def op_alignMPSubpsecs_fd(indat,minppm,maxppm,mode='o',initPars=None,ppmWeights=None):
    outdat,fs,phs=op_alignMPSubpsecs(indat,mode=mode,initPars=initPars,ppmWeights=ppmWeights,freq_range=[minppm,maxppm])
    return outdat,fs,phs

def op_alignrcvrs(indat,phasept=0,mode='w',coilcombos=None):
    if indat.flags['addedrcvrs']:
        raise Exception('ERROR: Receivers have already been combined! Aborting!')
        
    #To get best possible SNR, add the averages together (if it hasn't already been done):
    if not indat.dims['averages']>-1:
        av=op_averaging(indat)
    else:
        av=indat.copy()
    # also, for best results, we will combine all subspectra:
    if coilcombos is None:
        if indat.flags['isFourSteps']:
            av=op_fourStepCombine(av)
        if indat.dims['subSpecs']>-1:
            av=op_combinesubspecs(av,'summ')
            
    if coilcombos is None:
        # Code was repeated in Matlab rather than calling function
        coilcombos=op_getcoilcombos(av,phasept=phasept,mode=mode)
    phs=coilcombos['phs']
    # sigs does not seem to be normalized in alignrcvrs the way it is in addrcvrs
    # (there is some code to normalize by max sig but this is commented out). In Matlab,
    # op_getcoilcombos already normalized by max sig and this info can't be easily
    # recovered to match the lack of normalization here in op_alignrcvrs. So I've
    # removed the normalization in op_getcoilcombos and will have to add an
    # appropriate normalization back in after any function that calls it. (I'm
    # not sure why these different functions had different normalization approaches
    # in the first place)
    sigs=coilcombos['sigs']
    
    # now expand these matrices to match the size of indat.fids for 
    # multiplication. Each coil has a different phase and amplitude
    ph=np.ones(indat.sz)
    sig=np.ones(indat.sz)
    slice1=[slice(None)]*indat.fids.ndim
    for nct in range(indat.sz[indat.dims['coils']]):
        slice1[indat.dims['coils']]=nct
        ph[tuple(slice1)]=phs[nct]
        sig[tuple(slice1)]=sigs[nct]
    
    # now apply the phases by multiplying the data by exp(-i*ph);
    fids=indat.fids*np.exp(-1j*ph*np.pi/180)
    outdat=indat.copy()
    outdat.fids=fids
    #Make the coilcombos structure:
    coilcombos={'phs':phs,'sigs':sigs}
    return outdat,coilcombos

def op_alignScans(inref, infloat, tmax=0.5, mode='fp', freq_range=None, initPars=None):
    # Based on the Matlab code. The idea here is that we have parameters operating
    # on complex data, but least squares calculation for minimizing will do 
    # strange things with complex data. So one alternative is to concatenate the
    # real and imaginary parts of the data into a single vector, twice as long
    # and compare all components.
    def freqShiftComplexNest(in2,f):
        t=np.linspace(0,len(in2)*infloat.dwelltime,len(in2)+1)[:-1]#np.r_[0:(len(in2))*infloat.dwelltime:infloat.dwelltime]
        fid1=in2.flatten()
        y=fid1*np.exp(-1j*t.T*f*2*np.pi)
        y=np.r_[np.real(y),np.imag(y)]
        return y
    def phaseShiftComplexNest(in2,p):
        fid1=in2.flatten()
        y=add_phase(fid1,p)
        y=np.r_[np.real(y),np.imag(y)]
        return y
    def freqPhaseShiftComplexNest(in2,f,p):
        t=np.linspace(0,len(in2)*infloat.dwelltime,len(in2)+1)[:-1]#np.r_[0:(len(in2))*infloat.dwelltime:infloat.dwelltime]
        fid1=in2.flatten()
        y=add_phase(fid1*np.exp(-1j*t.T*f*2*np.pi),p)
        y=np.r_[np.real(y),np.imag(y)]
        return y
    
    # Jamie runs a bunch of checks and I'm not sure whether they are all necessary
    # However, I'm also running into problems because of the way that the flags
    # work (if there was no 'coils' dimension initially then flags['addedrcvrs']
    # will be False even though there is only 1 coil dimension). Similarly for
    # averages. I think that this should probably be fixed in FID.__init__ so 
    # that the initial flags match up with the dimension we start with, but
    # it's also possible to fix it here.
    # I've been working on more generalizable functions that accept multi-dimensional
    # input and work regardless which axis things are applied to. But I don't 
    # think that makes sense in this case because it's a fitting function and
    # everything needs to get into 1D, including concatenating real and imag parts
    # So I think the initial check should be whether the data are 1D and other
    # checks only run if that fails, as a way to provide extra info about what
    # needs to be done (averaging, coil combination, etc.)
    if not (inref.ndim==1 and infloat.ndim==1):
        if not inref.flags['addedrcvrs'] or not infloat.flags['addedrcvrs']:
            raise TypeError('ERROR: Only makes sense to do this after channels have been combined using op_addrcvrs. ABORTING!!')
        if not inref.flags['averaged'] or not infloat.flags['averaged']:
            raise TypeError('ERROR: Only makes sense to do this after you have combined averages using op_averaging. ABORTING!!')
        if not inref.flags['isFourSteps'] or not infloat.flags['isFourSteps']:
            raise TypeError('ERROR: Only makes sense to do this after you have performed op_fourStepCombine. ABORTING!!')
        raise Exception('ERROR: Data appear to have more than 1 dimension. ABORTING!! \ninref.ndim={:d} and infloat.ndim={:d}'.format(inref.ndim, infloat.ndim))
    if initPars is None:
        initPars=[0]*len(mode)
    if tmax>inref.t[-1]:
        tmax=inref.t[-1]
    # curve_fit needs an array of floats to fit to, so putting the real and imaginary components
    # into one longer real-valued vector for fitting
    # This seems to create problems. Not sure if it's how I've defined ppm in terms of center frequence
    # and sweep with, or if it's because the t limits is being applied after this, but it's just
    # not quite right, it seems.
    if freq_range is not None:
        inref_range=op_freqrange(inref, freq_range[0], freq_range[1])
        infloat_range=op_freqrange(infloat, freq_range[0], freq_range[1])
    else:
        inref_range=inref
        infloat_range=infloat
    baseref=np.r_[np.real(inref_range.fids[np.logical_and(inref_range.t>=0,inref_range.t<tmax)]),np.imag(inref_range.fids[np.logical_and(inref_range.t>=0,inref_range.t<tmax)])]
    if mode=='f':
        # But here send in complex-valued in1.fids for the same t range because
        # the real and imaginary parts will be separated after frq and ph adjustment
        parsFit,pcov=curve_fit(freqShiftComplexNest,infloat_range.fids[np.logical_and(infloat_range.t>=0,infloat_range.t<tmax)].squeeze(),baseref.squeeze(),initPars, maxfev=5000)
        frq=parsFit[0]; ph=0;
    elif mode=='p':
        parsFit,pcov=curve_fit(phaseShiftComplexNest,infloat_range.fids[np.logical_and(infloat_range.t>=0,infloat_range.t<tmax)].squeeze(),baseref.squeeze(),initPars, maxfev=5000)
        ph=parsFit[0]; frq=0;
    elif mode=='fp' or mode=='pf':
        parsFit,pcov=curve_fit(freqPhaseShiftComplexNest,infloat_range.fids[np.logical_and(infloat_range.t>=0,infloat_range.t<tmax)].squeeze(),baseref.squeeze(),initPars, maxfev=5000)
        frq=parsFit[0]; ph=parsFit[1];
    else:
        raise TypeError('ERROR: unrecognized mode. Please enter either "f", "p" or "fp"')
    # Note: don't need separate functions for final fid calculation because you
    # can just use existing op_freqshift and op_addphase functions with the 
    # fitted parameters (or defaults in 'f' and 'p' modes, as appropriate).
    # And you want to apply to the full spectrum, not just selected frequency range
    out1=op_addphase(op_freqshift(infloat,frq),ph)
    return out1, ph, frq

def op_alignScans_fd(inref, infloat, fmin, fmax, tmax=0.5, mode='fp'):
    out1,ph,frq=op_alignScans(inref, infloat, tmax=tmax, mode=mode, freq_range=[fmin,fmax])
    return out1, ph, frq

def op_ampScale(indat1,A):
    """
    Scale the amplitude of a spectrum by factor A. This function exists for 
    those replicating Matlab work. However, the __mult__ method is defined for 
    the FID object so that users can just call A*indat1 directly.

    Parameters
    ----------
    indat1 : FID object
        Spectrum to scale.
    A : float
        Amplitude scaling factor

    Returns
    -------
    outdat : FID object
        The output resulting from amplitude scaling.
    """
    return A*indat1

def op_arsos(indat,domain='t'):
    # The Matlab version seems to assume that there is only an averages dimension
    # and it is the second dimension. Here, the function is generalized, but
    # presumably subspecs should be combined, etc.
    if indat.flags['averaged'] or indat.dims['averages']==-1 or indat.averages<2:
        raise Exception('ERROR: Averaging has already been performed. Aborting!')
        
    outdat=indat.copy()
    if domain=='t':
        # Matlab sorts the real and imaginary parts separately. What?!
        fids=np.sort(np.real(indat.fids),axis=indat.dims['averages'])+1j*np.sort(np.imag(indat.fids),axis=indat.dims['averages'])
        # Do a DC shift correction using the last 25% of the time domain data
        tminIndex=np.ceil(np.median(0.75*len(indat.t)))
        medianReal=np.median(np.real(fids),indat.dims['averages'])
        medianImag=np.median(np.imag(fids),indat.dims['averages'])
        partslice=[slice(None)]*indat.ndim
        avslice=[slice(None)]*indat.ndim
        medianslice=[slice(None)]*indat.ndim
        for nct in range(indat.sz[indat.dims['averages']]):
            avslice[0]=slice(tminIndex,-1)
            medianslice[0]=slice(tminIndex,-1)
            avslice[indat.dims['averages']]=nct
            partslice[indat.dims['averages']]=nct
            DC0real=np.mean(np.real(fids[tuple(avslice)]),axis=0)-np.mean(medianReal[tuple(medianslice)],axis=0)
            DC0imag=np.mean(np.imag(fids[tuple(avslice)]),axis=0)-np.mean(medianImag[tuple(medianslice)],axis=0)
            fids[tuple(partslice)]=fids[tuple(partslice)]-(DC0real+1j*DC0imag)
        outdat.fids=fids
    elif domain=='f':
        specs=np.sort(np.real(indat.specs),axis=indat.dims['averages'])+1j*np.sort(np.imag(indat.specs),axis=indat.dims['averages'])
        # Do a DC shift correction using everything below 0 ppm and everything
        # above 10 ppm.
        ppmrange=np.nonzero((indat.ppm<0) | (indat.ppm>10))[0]
        medianReal=np.median(np.real(specs),indat.dims['averages'])
        medianImag=np.median(np.imag(specs),indat.dims['averages'])
        partslice=[slice(None)]*indat.ndim
        avslice=[slice(None)]*indat.ndim
        medianslice=[slice(None)]*indat.ndim
        for nct in range(indat.sz[indat.dims['averages']]):
            avslice[0]=slice(ppmrange[0],ppmrange[-1])
            medianslice[0]=slice(ppmrange[0],ppmrange[-1])
            avslice[indat.dims['averages']]=nct
            partslice[indat.dims['averages']]=nct
            DC0real=np.mean(np.real(specs[tuple(avslice)]),axis=0)-np.mean(medianReal[tuple(medianslice)],axis=0)
            DC0imag=np.mean(np.imag(specs[tuple(avslice)]),axis=0)-np.mean(medianImag[tuple(medianslice)],axis=0)
            specs[tuple(partslice)]=specs[tuple(partslice)]-(DC0real+1j*DC0imag)
        outdat.specs=specs
    else:
        print("Valid domains are 't' and 'f'. Returning input unaltered")
    return outdat

def op_autophase(indat,ppmmin,ppmmax,ph=0,dimNum=None):
    # It is possible to generalize this function to allow multiple dimensions, 
    # but it is less clear what the most sensible autophase decision is. Subsequent
    # averages should generally have the same phase correction, for example.
    # There may be some small differences in phase but these could be dealt with
    # via op_alignAverages(). So I currently just followed the Matlab case that
    # assumes 1D or allows autophasing of 1 part of a subSpectra. Note that op_addphase
    # does allow broadcasting so you could allow different autophases to be
    # calculated on the n last dimensions and then add them on
    if indat.dims['coils']>-1:
        raise Exception('ERROR: Cannot operate on data with multiple coils! ABORTING!!')
    if indat.dims['averages']>-1:
        raise Exception('ERROR: Cannot operate on data with multiple averages! ABORTING!!')
    if indat.dims['extras']>-1:
        raise Exception('ERROR: Cannot operate on data with extras dimension! ABORTING!!')
    if indat.dims['subSpecs']>-1:
        if dimNum is None:
            print('WARNING: No dimNum entered for subspectra selection. Autophasing the 0th subspec')
            dimNum=0
    if not indat.flags['zeropadded']:
        in_zp=op_zeropad(indat,10)
    else:
        in_zp=indat.copy()
    in_zp=op_freqrange(in_zp,ppmmin,ppmmax)
    if dimNum is None:
        ppmindex=np.argmax(np.abs(in_zp.specs))
        ph0=-1*np.angle(in_zp.specs[ppmindex])*180/np.pi
    else:
        # Only one dimension is autophased in the subspec case
        ppmindex=np.argmax(np.abs(in_zp.specs[:,dimNum]))
        ph0=-1*np.angle(in_zp.specs[ppmindex,dimNum])*180/np.pi
    phShft=ph+ph0
    outdat=op_addphase(indat,phShft)
    return outdat,phShft

def op_averaging(indat):
    if indat.flags['averaged'] or indat.averages<2 or indat.dims['averages']==-1:
        print('WARNING: No averages found. Returning input without modification!')
        outdat=indat.copy()
    else:
        outdat=indat.copy()
        # average spectrum along averages dimension (previously it was a sum and divide by shape, but why?)
        outdat.fids=np.mean(indat.fids,axis=indat.dims['averages']).squeeze()
        outdat._remove_dim_from_dict('averages')
        outdat.averages=1
        outdat.flags['averaged']=True
    return outdat

def op_blockAvg(indat,N):
    # Check if there are averages
    if indat.flags['averaged'] or indat.averages<2 or indat.dims['averages']==-1:
        print('WARNING: No averages found. Returning input without modification!')
        outdat=indat.copy()
    else:
        # Check if the number of averages is an even multiple of N. In the original
        # Matlab, this throws an error. Here, it only gives a warning because slicing
        # will just mean the last block will have fewer averages
        if np.remainder(indat.sz[indat.dims['averages']],N)!=0:
            print('WARNING: The number of averages {:d} does not divide evenly into N={:d}. The last grouping will be made up of fewer averages.'.format(indat.sz[indat.dims['averages']],N))
        avslice=[slice(None)]*indat.ndim
        tmpslice=[slice(None)]*indat.ndim
        Ntrans=int(np.ceil(indat.sz[indat.dims['averages']]/N))
        newsz=list(indat.sz)
        newsz[indat.dims['averages']]=Ntrans
        tmpfid=np.zeros(newsz,dtype=indat.fids.dtype)
        for nct in range(Ntrans):
            avslice[indat.dims['averages']]=slice(nct*N,(nct+1)*N)
            tmpslice[indat.dims['averages']]=nct
            tmpfid[tuple(tmpslice)]=np.mean(indat.fids[tuple(avslice)],axis=indat.dims['averages'])
        outdat=indat.copy()
        outdat.fids=tmpfid
        outdat.averages=Ntrans
    return outdat

def op_combineRcvrs(indat,inw):
    # First find the weights using the water unsuppressed data
    weights=op_getcoilcombos(inw,2,'h')
    weights['sigs']=weights['sigs']/np.max(weights['sigs'])
    # Now apply the weights to both the water unsuppressed and water suppressed
    #data, but don't combine the averages:
    out_presum=op_alignrcvrs(indat,2,'h',weights)
    outw_presum=op_alignrcvrs(inw,2,'h',weights)
    # Now apply the weights and combine the averages:
    outdat=op_addrcvrs(indat,2,'h',weights)
    outw=op_addrcvrs(inw,2,'h',weights)
    return outdat,outw,out_presum,outw_presum

def op_combinesubspecs(indat,mode):
    """
    Combine the subspectra in an acquisition either by addition or subtraction

    Parameters
    ----------
    indat : FID object
        Input spectrum.
    mode : str, 'diff' or 'summ'
        How to combine the data:
            -'diff' adds the subspectra together. This is counter-intuitive but
            the reason is that many "difference editing" sequences use phase
            cycling of the readout ADC to achieve "subtraction by addition".
            -'summ' performs a subtraction of the subspectra

    Returns
    -------
    outdat : FID object
        Output spectrum following combination of subspectra.

    """
    outdat=indat.copy()
    if indat.flags['subtracted']:
        raise Exception('ERROR: Subspectra have already been combined. Aborting!')
    if indat.flags['isFourSteps']:
        raise Exception('ERROR: Data with four steps must first be converted using op_fourStepCombine. Aborting!')
    if mode=='diff':
        # add the spectrum along the subSpecs dimension
        newfid=np.sum(indat.fids,axis=indat.dims['subSpecs'])/indat.sz[indat.dims['subSpecs']]
    elif mode=='summ':
        # Note that the assumption from Matlab is that there are only two subSpecs
        # but I did generalize this code to assume that even transients should 
        # be subtracted from odd
        sumslice1=[slice(None)]*indat.ndim
        sumslice1[indat.dims['subSpecs']]=slice(1,None,2)
        sumslice0=[slice(None)]*indat.ndim
        sumslice0[indat.dims['subSpecs']]=slice(0,None,2)
        newfid=np.sum(indat.fids[tuple(sumslice1)]-indat.fids[tuple(sumslice0)],axis=indat.dims['subSpecs'])/indat.sz[indat.dims['subSpecs']]
    outdat.fids=newfid
    outdat._remove_dim_from_dict('subSpecs')
    outdat.subSpecs=1
    outdat.averages=indat.averages/2
    outdat.flags['subtracted']=True
    return outdat

def op_complexConj(indat):
    """
    Take the complex conjugate of the data

    Parameters
    ----------
    indat : FID object
        Input data.

    Returns
    -------
    outdat : FID object
        Output following conjugation.

    """
    conjfid=np.conj(indat.fids)
    outdat=indat.copy()
    outdat.fids=conjfid
    return outdat

def op_concatAverages(indat1, indat2):
    """
    Concatenate two scans along the averages dimension. Two scans with 50 
    averages each will now look like a single scan with 100 averages. For
    looping, it is possible to enter None as the first input. In this case, the
    function will simply return the second input.

    Parameters
    ----------
    indat1 : FID object
        First input spectra to be concatenated.
    indat2 : TFID object
        Second input spectra to be concatenated.

    Returns
    -------
    outdat : FID object
        Output following concatenation of inputs along the averages dimension.
    """
    # I don't fully understand this case but Jamie includes it in Matlab for the
    # case of looping.
    if indat1 is None:
        outdat=indat2.copy()
    else:
        # Jamie has a check that both inputs have the same number of averages
        # (unless one or both input have no averages) but I don't understand 
        # why that restriction exists. I just need to expand_dims for singleton 
        # cases. The other dimensions have to match but np.concatenate will 
        # already fail if that is untrue
        if indat1.dims['averages']==-1 and indat2.dims['averages']==-1:
            avgdim=max(indat1.dims.values())+1
            fid1=np.expand_dims(indat1.fids,axis=avgdim)
            fid2=np.expand_dims(indat2.fids,axis=avgdim)
        elif indat1.dims['averages']==-1 and indat2.dims['averages']!=-1:
            fid1=np.expand_dims(indat1.fids,axis=indat2.dims['averages'])
            fid2=indat2.fids
            avgdim=indat2.dims['averages']
        elif indat1.dims['averages']==-1 and indat2.dims['averages']!=-1:
            fid2=np.expand_dims(indat2.fids,axis=indat1.dims['averages'])
            fid1=indat1.fids
            avgdim=indat1.dims['averages']
        else: # both dimensions have averages. Simplest case
            fid1=indat1.fids
            fid2=indat2.fids
            avgdim=indat1.dims['averages']
        outfid=np.concatenate((fid1,fid2),axis=avgdim)
        outdat=indat1.copy()
        outdat.fids=outfid
        outdat.dims['averages']=avgdim
        outdat.averages=indat1.averages+indat2.averages
        outdat.rawAverages=indat1.rawAverages+indat2.rawAverages
        outdat.flags['averaged']=False
        return outdat
    
def op_concatFreq(indat1,indat2,shift=None):
    # I don't understand the use case for this. Maybe for plotting but then
    # you should just deal with that in the plot. Function seems incomplete in
    # Matlab. Documentation hasn't been updated. Lots of concatAverages has 
    # been copied and commented out but not deleted.
    # Note that things are implemented quite differently than in Matlab because
    # of how ppm is defined by the spectral width and center frequency, rather
    # than being directly settable.
    if indat1.dims['t'] != indat2.dims['t']:
        # Really I want the actual 't' values to be the same, no? Otherwise, their
        # ppm values can't be easily concatenated alongside one another
        raise Exception('time/frequency dimension must be the same for both inputs.')
    dppm=np.abs(indat1.ppm[1]-indat1.ppm[0])
    if shift is None:
        newspec=np.concatenate((indat1.specs,indat2.specs),axis=indat1.dims['t'])
        # The ppm calculation has to work differently because of how I've set
        # up the setters in the FID object
        newspectralwidthppm=2*indat1.spectralwidthppm+dppm
        # using ppm[1]-ppm[0] instead of dppm because the signs will be consistent this way
        newcenterfreq=indat1.ppm[-1]+(indat1.ppm[1]-indat1.ppm[0])/2
    elif shift<indat1.spectralwidthppm:
        raise ValueError('ERROR: shift must be at least as large as the spectral width in ppm! ABORTING!!')
    else:
        # Make a dummy spectrum that is the same width as the desired spectral gap
        dummyspec=0*indat1.specs
        # as I interpret the Matlab code, shift is the ppm value where you want
        # the second spectrum to start. gap is then the distance between these two
        gap=shift-indat1.spectralwidthppm
        npts=np.int(np.ceil(gap/dppm))
        dummyspec=dummyspec[:npts,...]
        dummy_freqrange=npts*dppm # Note that you may end up with a shift slightly different than specified
        newspec=np.concatenate((indat1.specs,dummyspec),axis=indat1.dims['t'])
        newspec=np.concatenate((newspec,indat2.specs),axis=indat1.dims['t'])
        # The ppm calculation has to work differently because of how I've set
        # up the setters in the FID object
        newspectralwidthppm=indat1.spectralwidthppm+dummy_freqrange+indat2.spectralwidthppm
        if indat1.ppm[-1]<indat1.ppm[0]:
            newcenterfreq=indat1.ppm[-1]+newspectralwidthppm/2
        else:
            newcenterfreq=indat1.ppm[0]+newspectralwidthppm/2
    outdat=indat1.copy()
    outdat.specs=newspec
    outdat.spectralwidthppm=newspectralwidthppm
    outdat.center_freq_ppm=newcenterfreq
    # Matlab sets the averaged flag to false. Why? I think it's a holdover from
    # copying from op_concatAverages and doesn't belong. But not sure it matters 
    # since you don't really want to be doing new operations on this FID because 
    # it's time domain representation is completely messed up.
    outdat.flags['averaged']=False
    return outdat
    
def op_concatSubspecs(indat1,indat2):
    if indat1.dims['subSpecs'] != indat2.dims['subSpecs'] or indat1.dims['t'] != indat2.dims['t'] or indat1.dims['averages'] != indat2.dims['averages'] or indat1.dims['coils'] != indat2.dims['coils']:
        raise Exception('ERROR: subSpecs dimensions must be the same for both inputs')
    # if there is no subspecs dimension, make one. Since we already checked that
    # the dimensions are the same, we can do fewer cases than op_concatAverages
    if indat1.dims['subSpecs']==-1 and indat2.dims['subSpecs']==-1:
        ssdim=max(indat1.dims.values())+1
        fid1=np.expand_dims(indat1.fids,axis=ssdim)
        fid2=np.expand_dims(indat2.fids,axis=ssdim)
    else: # both dimensions have averages. Simplest case
        fid1=indat1.fids
        fid2=indat2.fids
        ssdim=indat1.dims['subSpecs']
    outdat=indat1.copy()
    outdat.fids=np.concatenate((fid1,fid2),axis=ssdim)
    outdat.dims['subSpecs']=ssdim
    outdat.subSpecs=indat1.subSpecs+indat2.subSpecs
    outdat.rawSubspecs=indat1.rawSubspecs+indat2.rawSubspecs
    outdat.flags['subtracted']=False
    return outdat

def op_filter(indat,lb):
    """
    Perform line broadening by multiplying the time domain signal by an 
    exponential decay function.  

    Parameters
    ----------
    indat : FID class
        input data.
    lb : float
        Line broadening factor in Hz.

    Returns
    -------
    outdat : FID class
        Output following alignment of averages.
    lor : Numpy array
        Exponential time domain filter that was applied
    """
    if lb==0:
        outdat=indat
        lor=None
    else:
        if indat.flags['filtered']:
            cont=input('WARNING:  Line Broadening has already been performed!  Continue anyway?  (y or n)')
            if cont=='y':
                fids=indat.fids.copy()
                t2=1/(np.pi*lb)
                # Create an exponential decay (lorentzian filter) and tile so it has same size as fids
                lor=np.exp(-1*indat['t']/t2)
                fil=np.tile(lor,list(fids.shape)[1:]+[1]).transpose([-1]+list(range(fids.ndim-1)))
                #Now multiply the data by the filter array.
                fids=fids*fil
                # Filling in the data structure and flags
                outdat=indat.copy()
                outdat['fids']=fids
                outdat.flags['writtentostruct']=True
                outdat.flags['filtered']=True
        else:
            outdat=indat
            lor=None
    return outdat,lor

def op_fourStepCombine(indat,mode=0):
    if not indat.flags['isFourSteps']:
        raise AttributeError('ERROR: requires a dataset with 4 subspecs as input!  Aborting!')
    if indat.sz[-1]!=4:
        raise TypeError('ERROR: final matrix dim must have length 4!!  Aborting!')
    # now make subspecs and subfids (This doesn't do anything to MEGA-PRESS
    # data, but it combines the SPECIAL iterations in MEGA-SPECIAL).
    sz=indat.sz
    reshapedFids=np.reshape(indat.fids,[np.prod(sz[:-2]),sz[-1]])
    sz[-1]=sz[-1]-2
    if mode==0:
        reshapedFids[:,0]=np.sum(reshapedFids[:,[0,1]],axis=1)
        reshapedFids[:,1]=np.sum(reshapedFids[:,[2,3]],axis=1)
    elif mode==1:
        reshapedFids[:,0]=np.diff(reshapedFids[:,[0,1]],axis=1)
        reshapedFids[:,1]=np.diff(reshapedFids[:,[2,3]],axis=1)
    elif mode==2:
        reshapedFids[:,0]=np.sum(reshapedFids[:,[0,2]],axis=1)
        reshapedFids[:,1]=np.sum(reshapedFids[:,[1,3]],axis=1)
    elif mode==3:
        reshapedFids[:,0]=np.diff(reshapedFids[:,[0,2]],axis=1)
        reshapedFids[:,1]=np.diff(reshapedFids[:,[1,3]],axis=1)
    else:
        raise ValueError('ERROR: mode not recognized. Value must be 0, 1, 2 or 3')
    fids=np.reshape(reshapedFids[:,[0,1]],sz)
    outdat=indat.copy()
    outdat.fids=fids/2  #Divide by 2 so that this is an averaging operation
    outdat.subSpecs=outdat.sz[outdat.dims['subSpecs']]
    outdat.flags['isFourSteps']=False
    return outdat

def op_freqrange(indat,ppmmin,ppmmax):
    fullspec=indat.specs.copy()
    outdat=indat.copy()
    indvals=np.logical_and(np.greater(indat.ppm,ppmmin),np.less(indat.ppm,ppmmax))
    outdat.specs=fullspec[indvals,...]
    # Need to redefine the ppm range, which is done by setting the center_freq_ppm
    # and the spectralwidth
    outdat.center_freq_ppm=ppmmin+(ppmmax-ppmmin)/2
    outdat.spectralwidthppm=np.abs(ppmmax-ppmmin)
    outdat.flags['freqranged']=True
    return outdat

def freqrange(inspec,ppm,ppmmin,ppmmax):
    # differs from op_freqrange in that that operates on a fid object, whereas
    # this only requires the frequency spectrum and ppm
    indvals=np.logical_and(np.greater(ppm,ppmmin),np.less(ppm,ppmmax))
    specpart=inspec[indvals,:]
    ppmpart=ppm[indvals]
    return ppmpart,specpart

def op_freqshift(indat,fshift):
    outdat=indat.copy()
    t=np.tile(indat.t,list(indat.sz[1:])+[1]).T
    outdat.fids=indat.fids*np.exp(-1j*t*fshift*2*np.pi)
    return outdat
    
def op_getcoilcombos(indat,phasept=0,mode='w'):
    """
    Finds the relative coil phases and amplitudes for coil data in indat. The
    result can be fed to op_addrcvrs for coil combination (although data generally
    need to be rephased after)

    Parameters
    ----------
    indat : FID object
        Input data with multiple receiver info. Note that Matlab fidA allows 
        you to enter a filename but assumes twix format. This differs from other
        processing functions, so I'm removing that option in Python.
    phasept : float, optional
        Point of fid to use for phase estimation and amplitude when mode='w'. 
        The default is 0.
    mode : char, optional
        Method for estiamting the coil weights and phases if not provided in 
        coilcombos. Can be:
            'w' - performs amplitude weighting of channels based on the max 
                signal of each coil channel
            'h' - performs amplitude weighting of channles based on the max signal 
                of each coil channel divided by the square of the noise in each 
                coil channel (as described by Hall et al. Neuroimage 2014).
            The default is 'w'.

    Returns
    -------
    coilcombos : dict with keys 'phs' and 'sigs'
        The vectors of the coil phases (in degrees) used for alignment and the 
        coil weights.

    """
    if indat.flags['addedrcvrs'] or indat.dims['coils']==-1 or indat.sz[indat.dims['coils']]==1:
        print('WARNING: Only one receiver channel found! Coil phase will be 0.0 and coil amplitude will be 1.0.')
        coilcombos={'phs':0, 'sigs':1}
    else:
        # Find the relative phases between the channels and populate the ph matrix
        # The use of the slice object here allows this to be done for the 'coils'
        # dimension regardless of the other dimensions in indat.dims
        whichpts=[0]*indat.ndim
        whichpts[indat.dims['coils']]=slice(None)
        whichpts[indat.dims['t']]=phasept
        # Jamie unwraps the phase in the Matlab version but only ever uses 
        # the angle, so I'm leaving the phase wrapped for simplicity. Since
        # phs is returned, it may be used for used later and would look 
        # different when plotted
        phs=np.angle(indat.fids[tuple(whichpts)])*180/np.pi
        if mode=='w':
            sigs=np.abs(indat.fids[tuple(whichpts)])
        elif mode=='h':
            whichpts[indat.dims['t']]=slice(None)
            S=np.max(np.abs(indat.fids[tuple(whichpts)]),axis=indat.dims['t'])
            # -100 is copied from the Matlab code, assuming this is in the noise for the fid.
            whichpts[indat.dims['t']]=slice(-100,None)
            N=np.std(indat.fids[tuple(whichpts)],axis=indat.dims['t'])
            sigs=S/(N**2)
        else:
            raise TypeError("ERROR: mode must have type 'w' or 'h'.")
        # In Matlab, op_getcoilcombos normalizes so that the max signal amplitude
        # is 1, whereas op_addrcvrs normalizes so the sum of the amplitudes is
        # 1 and op_alignrcvrs doesn't normalize at all (commented out). I am
        # therefore removing the normalization within the function (ie. the 
        # Python result for this function will differ from Matlab) and any
        # normalization will need to be added after the function call. It doesn't
        # really make sense to me that different normalization methods were being
        # used for very similar function calls anyway.
        #sigs=sigs/np.max(sigs)
        coilcombos={'phs':phs,'sigs':sigs}
    return coilcombos

def op_getPeakHeight(indat,ppmmin=1.8,ppmmax=2.2):
    datsize=indat.specs.shape
    peak_mask=np.where((indat.ppm>ppmmin) & (indat.ppm<ppmmax),1,0)
    peak_mask=np.tile(peak_mask,list(datsize[1:])+[1]).transpose([len(datsize)-1]+[d for d in range(len(datsize)-1)])
    peak_height=np.amax(peak_mask*np.abs(indat.specs),axis=indat.dims['t'])
    return peak_height
    
def op_gaussian(ppm,amp,fwhm,ppm0,base_off=0,ph0=0):
    if type(amp) is int:
        amp=[amp]
        fwhm=[fwhm]
        ppm0=[ppm0]
    # don't need to make baseline and ph0 into lists because assume same for all peaks
    c=[fv/2/np.sqrt(2*np.log(2)) for fv in fwhm]
    y=np.zeros([len(amp),len(ppm)])
    for act,aval in enumerate(amp):
        y[act,:]=np.exp(-1*(ppm-ppm0[act])**2/2/c[act]**2)
        # Scale it, add baseline, phase by ph0, and take the real part
        y[act,:]=np.real(add_phase(y[act,:]/np.amax(np.abs(y[act,:]))*aval+base_off,ph0))
    y=np.sum(y,axis=0)
    return y

def op_creFit(indat,ph0,ph1,ppmmin=2.9,ppmmax=3.15,disp_result=True):
    # Note that this function is not generalizable to multiple dimensions because
    # the fit must produce a 1D vector. If you have MRS data with spatial dimensions
    # you will need to select one or loop over them and fit each separately
    if indat.flags['isFourSteps']:
        raise Exception('ERROR: Must have combined subspecs in order to do this! ABORTING')
    if not indat.flags['averaged']:
        raise Exception('ERROR: Must have averaged in order to do this! ABORTING')
    if not indat.flags['addedrcvrs']:
        raise Exception('ERROR: Must have added receivers in order to do this! ABORTING')
    phased_spec=op_addphase(indat,ph0,ph1)
    ind_vals=np.logical_and(np.greater(indat.ppm,ppmmin),np.less(indat.ppm,ppmmax))
    ppm_part=indat.ppm[ind_vals]
    spec_part=phased_spec.specs[ind_vals]
    # amp, FWHM, ppm0, base_slope, base_off,ph0 (each as its own list, but only 1 element for 1 peak)
    base_slope=(np.real(spec_part[0])-np.real(spec_part[-1]))/(ppm_part[0]-ppm_part[-1])
    parsGuess=[np.amax(np.real(spec_part)),
              3/indat.txfreq,
              ppmmin+(ppmmax-ppmmin)/2,
              base_slope,
              np.real(spec_part[0])-base_slope*ppm_part[0],
              0]
    parsGuess=[list(eachel) for eachel in parsGuess]
    yGuess=op_lorentz_linbas(parsGuess,ppm_part)
    parsFit, pcov=new_curvefit(op_lorentz_linbas, ppm_part, np.real(spec_part), p0=parsGuess)
    yFit=op_lorentz_linbas(parsFit,ppm_part)
    if disp_result:
        f1,ax1=plt.subplots(1,1)
        for eachy,ystyle,ylab in zip([spec_part,yGuess,yFit],['-',':','-'],['data','guess','fit']):
            ax1.plot(ppm_part,eachy,fmt=ystyle,label=ylab)
        print('Fit: '+', '.join(['{:3.2f}'.format(eachpar) for eachpar in parsFit]))
        print('Guess: '+', '.join(['{:3.2f}'.format(eachpar) for eachpar in parsGuess]))
        # The original op_creFit acts as though it fits several peaks and displays
        # the area for each, but it is only actually set up to fit one peak
        area=parsFit[0]*parsFit[1]
        print('Area under the fitted curve is: {:3.2f}'.format(area))
    return parsFit

# def op_lorentz_linbas_fitting(ppm,allpars):
#     npars=6
#     npeaks=len(allpars//npars)
#     parList=[allpars[nct*npeaks:(nct+1)*npeaks] for nct in range(npars)]
#     return op_lorentz_linbas(parList,ppm)

# def make_curve_fit_compatible(funcnm):
#     # note that you still need to make the flat_list before using the curve_fit 
#     # function. I made flatten_parlist() for this since the list comprehension syntax is confusing
#     # Can I use an argument to the decorator to set npars??
#     @functools.wraps(funcnm)
#     def wrapper(ppm,*flat_list):
#         # put list of pars into single list
#         npars=6#len(pars)
#         n_els=len(flat_list)//npars
#         par_list=[list(flat_list[nct*n_els:(nct+1)*n_els]) for nct in range(npars)]
#         yvals=funcnm(par_list,ppm)
#         return yvals
#         # put output back into separated list format if you have it?? Not sure
#         # this is possible because return is just the yvals. pars come from curve_fit
#     return wrapper

def new_curvefit(funcnm,xdata,ydata,parlist_of_lists,*args,**kwargs):
    # I don't think that you even need funcnm here, at least not like this. You just literally want something
    # that you can call and it always makes the stuff you need to reformat for curvefit
    npars=len(parlist_of_lists)
    npeaks=len(parlist_of_lists[0])
    flatlist=[parval for eachpar in parlist_of_lists for parval in eachpar]
    parsFit, pcov=curve_fit(funcnm, xdata, ydata, flatlist,*args,**kwargs)
    reformat_parsFit=[parsFit[nct*npeaks:(nct+1)*npeaks] for nct in range(npars)]
    #probably should reformat pcov as well but I never pay attention to what it looks like
    return reformat_parsFit,pcov

def op_lorentz_linbas(pars,ppm):
    # This function replicates the form of the Matlab code, with the arguments
    # in the same order. However, scipy.optimize.curve_fit requires the arguments
    # in a different form f(x,...) where x is the independent variable and the 
    # parameters to fit are the separate remaining arguments (ie. not in a list)
    # To fit this function, use new_curvefit.
    if len(pars)<3:
        raise ValueError('ERROR: pars must have length of at least 3 for amplitude, FWHM and center frequency.')
    # Assume all values are entered. Can fix defaults later with if statements
    [amp,fwhm,ppm0,base_slope,base_off,ph0]=pars
    if type(amp) is not list:
        for eachvar in [amp,fwhm,ppm0,base_slope,base_off,ph0]:
            eachvar=[eachvar]
    hwhm=[fv/2 for fv in fwhm]
    y=np.zeros([len(amp),len(ppm)])
    for act,aval in enumerate(amp):
        y[act,:]=np.sqrt(2/np.pi)*(hwhm[act]-1j*(ppm-ppm0[act]))/(hwhm[act]**2+(ppm-ppm0[act])**2)
        # Scale it, add baseline, phase by ph0, and take the real part
        bas=base_off[act]+base_slope[act]*ppm
        y[act,:]=np.real(add_phase(y[act,:]/np.amax(np.abs(y[act,:]))*aval+bas,ph0[act]))
    y=np.sum(y,axis=0)
    return y.squeeze()

def op_lorentz(ppm,amp,fwhm,ppm0,base_off=0,ph0=0):
    if type(amp) is not list:
        amp=[amp]
        fwhm=[fwhm]
        ppm0=[ppm0]
    # don't need to make baseline and ph0 into lists because assume same for all peaks
    hwhm=[fv/2 for fv in fwhm]
    y=np.zeros([len(amp),len(ppm)])
    for act,aval in enumerate(amp):
        y[act,:]=np.sqrt(2/np.pi)*(hwhm[act]-1j*(ppm-ppm0[act]))/(hwhm[act]**2+(ppm-ppm0[act])**2)
        # Scale it, add baseline, phase by ph0, and take the real part
        y[act,:]=np.real(add_phase(y[act,:]/np.amax(np.abs(y[act,:]))*aval+base_off,ph0))
    y=np.sum(y,axis=0)
    return y.squeeze()

def op_median(indat):
    if indat.flags['averaged'] or indat.dims['averages']==-1 or indat.averages<2:
        print('ERROR:  Averaging has already been performed!  Aborting!')
        outdat=indat
    else:
        outdat=indat.copy()
        # add the spectrum along the averages dimension
        outdat.fids=np.median(indat.fids,axis=indat.dims['averages']).squeeze()
        # change dims variable and update flags
        outdat._remove_dim_from_dict('averages')
        outdat.averages=1
        outdat.flags['averaged']=True
        outdat.flags['writtentostruct']=True
    return outdat
    
def op_peakFit(inspec,ppm,amp,fwhm,ppm0,base_off,ph0,ppmmin=0,ppmmax=4.2):
    ppmrange,specrange=freqrange(inspec,ppm,ppmmin,ppmmax)
    specrange=np.real(specrange.squeeze())
    parsGuess=[amp,fwhm,ppm0,base_off,ph0]
    lb=[0,1e-5,ppmmin,-1*np.amax(inspec)/2,-np.pi]
    ub=[2*np.amax(inspec),0.5,ppmmax,np.amax(inspec)/2,np.pi]
    yGuess=op_lorentz(ppm, amp, fwhm, ppm0, base_off, ph0)
    parsFit, pcov=curve_fit(op_lorentz, ppmrange, specrange, p0=parsGuess, bounds=[lb,ub])
    yFit=op_lorentz(ppm,*parsFit)
    return parsFit,yFit,yGuess
    
def op_multi_peakFit(inspec,ppm,amp,fwhm,ppm0,base_off,ph0,ppmmin=0,ppmmax=4.2,peaktype='lorentz'):
    ppmrange,specrange=freqrange(inspec,ppm,ppmmin,ppmmax)
    specrange=np.real(specrange.squeeze())
    # Since curve_fit needs a single vector of all variables to fit, I need to
    # add together the lists of amplitudes, fwhm, etc. Will use a wrapping function
    # to unpack back into list form to sent to the fitting function
    namp=len(amp)
    parsGuess=amp+fwhm+ppm0+[base_off,ph0]
    lb=[0]*len(amp)+[1e-4]*len(amp)+[ppmmin]*len(amp)+[-1*np.amax(inspec)/2,-np.pi]
    ub=[2*np.amax(inspec)]*len(amp)+[0.4]*len(amp)+[ppmmax]*len(amp)+[np.amax(inspec)/2,np.pi]
    #ub[16]=0.2
    #ub[17]=0.2
    def unpack_vars(ppm1,*varlist):
        amplist=list(varlist[:namp])
        fwhmlist=list(varlist[namp:2*namp])
        ppm0list=list(varlist[2*namp:3*namp])
        baseval=varlist[-2]
        ph0val=varlist[-1]
        if peaktype=='lorentz':
            peak_fit=op_lorentz(ppm1,amplist,fwhmlist,ppm0list,baseval,ph0val)
        elif peaktype=='gauss':
            peak_fit=op_gaussian(ppm1,amplist,fwhmlist,ppm0list,baseval,ph0val)
        else:
            raise TypeError("Variable peaktype must be either 'lorentz' or 'gauss'")
        return peak_fit
    parsFit, pcov=curve_fit(unpack_vars, ppmrange, specrange, p0=parsGuess, bounds=[lb,ub])
    yFit=unpack_vars(ppm,*parsFit)
    parsDict={'amps':parsFit[:namp],'fwhms':parsFit[namp:2*namp],'ppm0s':parsFit[2*namp:3*namp],'base_off':parsFit[-2],'ph0':parsFit[-1]}
    return parsDict,yFit

def op_ppmref(indat,ppmmin,ppmmax,ppmrefval,dimNum=0,zpfact=10):
    # zeropad if it's not already done
    if not indat.flags['zeropadded']:
        in_zp=op_zeropad(indat,zpfact)
    else:
        print('Data already zeropadded. Using existing zero padding.')
        in_zp=indat.copy()
    # find the ppm of the maximum peak magnitude within a given range
    masked_spec=np.asarray((in_zp.ppm>ppmmin)*(in_zp.ppm<ppmmax))*np.abs(in_zp.specs[:,dimNum])
    ppmindex=np.argmax(masked_spec)
    # Jamie has an extra step here. Not sure it's necessary
    ppmmax=in_zp.ppm[ppmindex]
    frqshift=(ppmmax-ppmrefval)*indat.txfreq/1e6
    outdat=op_freqshift(indat, frqshift)
    return outdat,frqshift

def op_rm_bad_averages(indat,nsd=3,which_domain='t'):
    which_domain=which_domain.lower()
    if indat.flags['averaged']:
        print('ERROR:  Averaging has already been performed!  Aborting!')
        outdat=indat
    elif not indat.flags['addedrcvrs']:
        print('ERROR:  Receivers should be combined first!  Aborting!')
        outdat=indat
    else:
        #first, make a metric by subtracting all averages from the first average, 
        #and then taking the sum of all the spectral points.  
        if indat.dims['subSpecs']>-1:
            ss=indat.sz[indat.dims['subSpecs']]
        else:
            ss=0
        if which_domain=='t':
            infilt=indat.copy()
            tmax=0.4
        elif which_domain=='f':
            filt=10
            infilt=op_filter(indat,filt)
        # not sure why this is a median, but it's like that in the Matlab code 
        # and a similar call to op_averaging(infilt) is commented out
        inavg=op_median(infilt)
        # doing this differently than Matlab to avoid loops. Tile the average to make it the same size as before
        avgdim=infilt.dims['averages']
        repvec=[1]*infilt.fids.ndim
        repvec[avgdim]=infilt.fids.shape[avgdim]
        avgfids=np.tile(np.expand_dims(inavg.fids,axis=avgdim),repvec)
        trange=(infilt.t>=0) * (infilt.t<=tmax)
        if which_domain=='t':
            metric=np.sum((np.real(infilt.fids[trange,:,:])-np.real(inavg.fids[trange,:,:]))**2,axis=0)
        elif which_domain=='f':
            metric=np.sum((np.real(infilt.specs[trange,:,:])-np.real(inavg.specs[trange,:,:]))**2,axis=0)
        #find the average and standard deviation of the metric
        avg1=np.mean(metric,axis=0)
        sd1=np.std(metric,axis=0)
        
        #Now z-transform the metric  
        zmetric=(metric-avg1)/sd1
        
        P=np.zeros([ss,zmetric.shape[0]])
        f1,ax1=plt.subplots(1,ss)
        # more in text file
        for m in range(ss):
            P[m,:]=np.polyfit(range(indat.sz[indat.dims['averages']]),zmetric[:,m],deg=2)
            ax1[m].plot(np.r_[:indat.sz[indat.dims['averages']]],zmetric[:,m],'.',
                        np.r_[:indat.sz[indat.dims['averages']]],np.polyval(P[m,:],np.r_[:indat.sz[indat.dims['averages']]]),
                        np.r_[:indat.sz[indat.dims['averages']]],np.polyval(P[m,:],np.r_[:indat.sz[indat.dims['averages']]]).T+nsd,':')
            ax1[m].set_xlabel('Scan Number')
            ax1[m].set_ylable('Unlikeness Metric z-score')
            ax1[m].set_title('Metric for rejection of motion corrupted scans')
        # Now make a mask for locations more than nsd from the mean
        mask=np.zeros([zmetric.shape[0],ss])
        for m in range(ss):
            mask[:,m]=zmetric[:,m]>(np.polyval(P[m,:],np.r_[:indat.sz[indat.dims['averages']]]+nsd))
            
        #Unfortunately, if one average is corrupted, then all of the subspecs
        #corresponding to that average have to be thrown away.  Therefore, take the
        #minimum intensity projection along the subspecs dimension to find out
        #which averages contain at least one corrupted subspec:
        if mask.shape[1]>1:
            mask=(np.sum(mask,axis=1)>0)
        #Now the corrupted and uncorrupted average numbers are given by
        badAverages=np.nonzero(mask)[0]
        goodAverages=np.nonzero(1-mask)[0]
        # Make new fids array with only good averages
        outdat=indat.copy()
        outdat.fids=indat.fit[:,goodAverages,:,:]
        outdat.averages=len(goodAverages)*indat.rawSubspecs
        outdat.flags['writtentostruct']=1
    return outdat,metric,badAverages

def op_takesubspec(indat,idx):
    if indat.flags['subtracted']:
        raise Exception('ERROR: Subspectra have already been combined. ABORTING!')
    if indat.dims['subSpecs']>-1:
        raise Exception('ERROR: There are no subspectra in this dataset. ABORTING!')
    subspec_slice=[slice(None)]*indat.ndim
    subspec_slice[indat.dims['subSpecs']]=idx
    # subspec dimension should be automatically removed from outdat.dims by __getitem__
    outdat=indat[tuple(subspec_slice)]
    outdat.subSpecs=1
    return outdat

def op_zeropad(indat,zpfact):
    outdat=indat.copy()
    if indat.flags['zeropadded']:
        cflag=input('Warning: zero padding has already been performed. Continue? (y or n): ')
        if cflag.lower()=='y':
            # creatine peak for water-suppressed data
            newsz=list(indat.sz)
            newsz[0]=zpfact*indat.sz[0]
            outdat.fids=np.zeros(newsz,dtype=indat.fids.dtype)
            outdat.fids[:indat.sz[0],...]=indat.fids
    else:
        newsz=list(indat.sz)
        newsz[0]=zpfact*indat.sz[0]
        outdat.fids=np.zeros(newsz,dtype=indat.fids.dtype)
        outdat.fids[:indat.sz[0],...]=indat.fids
    # Note that t, dwelltime, _ppmmin, ppmrange, etc are all calculated from 
    # spectralwidth, which should be unchanged here, and len(specs), which will
    # be adjusted. So no need to recalculate.
    outdat.flags['zeropadded']=True
    return outdat

def op_plotspec(indat,xlims=[4.5,0],xlab='Chemical Shift (ppm)',ylab='Signal',title='',plotax=None):
    # Need to update to deal with multiple averages and other possible dimensions, as in Matlab op_plotspec
    if plotax is None:
        [f1,plotax]=plt.subplots(1,1)
    # Two cases: a list of FID objects, or a single FID object that may have 2+dimensions
    if type(indat)==list:
        # Check that every entry in the list is of effective size 1
        if any([eachit.fids.ndim>2 for eachit in indat]) or any([eachit.fids.ndim==2 and (not (1 in eachit.sz)) for eachit in indat]):
            print("List entries cannot have more than 1 dimension. Did you forget to average?")
        else:
            for eachit in indat:
                plotax.plot(eachit.ppm,np.real(eachit.specs))
            plotax.set_xlim(xlims)
            plotax.set_xlabel(xlab)
            plotax.set_ylabel(ylab)
            plotax.set_title(title)
    else:
        if indat.fids.ndim>2:
            print("Cannot plot for more than two dimensions")
        else:
            plotax.plot(indat.ppm,np.real(indat.specs))
            plotax.set_xlim(xlims)
            plotax.set_xlabel(xlab)
            plotax.set_ylabel(ylab)
            plotax.set_title(title)

if __name__ == '__main__':
    """
    for debugging
    """
    import fidA_io as fio
    import os
    # pname='/Users/nearlabmacbook1/Documents/BrukerData/SchuurmansMice'
    # fct=2
    # with open(os.path.join(pname,'flist_smallvox_right')) as f:
    #     ftmp=f.readlines()
    #     fright=[fn.strip() for fn in ftmp if fn.startswith('2024')]
    # with open(os.path.join(pname,'flist_smallvox_left')) as f:
    #     ftmp=f.readlines()
    #     fleft=[fn.strip() for fn in ftmp if fn.startswith('2024')]
    # out_left,ref_left,info_left=fio.io_loadspec_bruk(os.path.join(pname,fleft[fct]),try_raw=False)
    # out_right,ref_right,info_right=fio.io_loadspec_bruk(os.path.join(pname,fright[fct]),try_raw=False)
    # f1,ax1=plt.subplots(1,2)
    # out_left.plot_spec(plotax=ax1[0])
    # out_right.plot_spec(plotax=ax1[0])
    # new_right, ph1, frq1=op_align_scans(out_left, out_right)
    # out_left.plot_spec(plotax=ax1[1])
    # new_right.plot_spec(plotax=ax1[1])
    
    pname='/Users/nearlabmacbook1/Documents/BrukerData/FUS_pentobarbital/20240916_133721_SKWU1A_Sept16_2024_RK50_HL_SKWU1A_Sept16_20_1_4'
    fid1=fio.io_loadspec_bruk(os.path.join(pname,'5','rawdata.job0'))
    outdat,fids_presum,specs_presum,coilcombos=op_addrcvrs(fid1)