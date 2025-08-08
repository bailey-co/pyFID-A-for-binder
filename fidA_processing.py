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
GAMMAP=42.577

def add_phase(invec,added_phase):
    """
    Add equal amounts of complex phase to each point of a vector (note this 
    function operates on a vector rather than a fid object. To operate on the
    fid object, use 'op_addphase').

    Parameters
    ----------
    invec : input vector
    added_phase : float
        Amount of phase (in degrees) to add.

    Returns
    -------
    output vector
        0th order phased version of the input.

    """
    return invec*np.exp(1j*added_phase*np.pi/180)

def add_phase1(invec,ppm,timeShift,ppm0=4.65,B0=7):
    """
    Add first order phase to a spectrum (added phase in linearly dependent on 
    frequency). This function operates on  vector, not a fid-A object. For a
    phase shifting function that operates on a fid-A object, see 'op_addphase'

    Parameters
    ----------
    invec : input vector (spectrum)
    ppm : 1D vector
        Frequency scale (ppm) corresponding to the vector.
    timeShift : float
        Amount of 1st order phase shift (specified as horizontal shift in 
        seconds in the time domain).
    ppm0 : float, optional
        The frequency "origin" (in ppm) of the 1st order phase shift (this
        point will undergo 0 phase shift). The default is 4.65.
    B0 : float, optional
        Magnetic field strength in Tesla (needed to convert ppm to Hz). The default is 7.

    Returns
    -------
    phased_spec : output vector
        1st order phased version of the input.

    """
    f=np.expand_dims((ppm-ppm0)*GAMMAP*B0,axis=tuple(range(1,invec.ndim))) #untested but I think this should work for multidimensional vectors with 1D ppm
    f=np.tile(f,[1]+list(invec.shape[1:]))
    # note that f should be in Hz and timeshift in s, so multiplyling them 
    # gives result in cycles. Multiply by 2*pi to get phase in radians
    phased_spec=invec*np.exp(-1j*f*timeShift*2*np.pi)
    return phased_spec

def op_add_noise(indat,sdnoise):
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
    outdat : FID object.
        output data with noise added
    """
    outdat=indat.copy()
    if type(sdnoise) is np.ndarray:
        noisevec=sdnoise
    else:
        noisevec=sdnoise*np.random.randn(indat.sz)+1j*sdnoise*np.random.randn(indat.sz)
    outdat.fids=indat.fids+noisevec
    return outdat,noisevec

def op_addphase(indat,ph0,ph1=0,ppm0=4.65,suppressPlot=True):
    """
    Add zero and first order phase to the spectrum of an FID object.

    Parameters
    ----------
    indat : FID object
        input data.
    ph0 : float
        0th order phase correction in degrees.
    ph1 : float, optional
        1st order phase correction in seconds. The default is 0.
    ppm0 : float, optional
        frequency reference point in ppm. The default is 4.65.
    suppressPlot : boolean, optional
        flag to determine whether figure is suppressed. The default is True

    Returns
    -------
    outdat : FID object
        phase-adjusted output spectrum.

    """
    outdat=indat.copy()
    outdat.fids=indat.fids*np.exp(1j*ph0*np.pi/180)
    outdat.added_ph0=indat.added_ph0+ph0
    #Now add 1st-order phase
    newspec=add_phase1(outdat.specs,indat.ppm,ph1,ppm0,indat.Bo)
    outdat.set_fid_from_specs(newspec)
    outdat.added_ph1=indat.added_ph1+ph1
    if len(outdat.specs.shape)<3 and not suppressPlot:
        plt.figure()
        outdat.plot_spec()
    return outdat

def op_addphaseSubspec(indat,ph0):
    """
    For spectra with two subspectra, add zero order phase to the second 
    subspectra in a dataset. For example, the edit-on spectrum of a mega-press 
    acquisition.    

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
        raise TypeError('ERROR: Can not operate on data with multilple coils!  ABORTING!!')
    elif indat.dims['averages']>=0:
        raise TypeError('ERROR: Can not operate on data with multiple averages!  ABORTING!!')
    elif indat.dims['subSpecs']<0:
        raise TypeError('ERROR:  Can not operate on data with no Subspecs!  ABORTING!!')
    if indat.sz[indat.dims['subSpecs']]!=2:
        raise TypeError('ERROR:  Input spectrum must have two subspecs!  ABORTING!!')
    outdat=indat.copy()
    outdat.fids[:,1]=outdat.fids[:,1]*np.exp(1j*ph0*np.pi/180)
    return outdat
    
def op_addrcvrs(indat,phasept=0,mode='w',coilcombos=None):
    if indat.flags['addedrcvrs'] or indat.dims['coils']==-1:
        print('WARNING:  Only one receiver channel found!  Returning input without modification!')
        outdat=indat.copy()
        outdat.flags['addedrcvrs']=1
        fids_presum=indat.fids
        specs_presum=indat.specs
        coilcombos={'phs':0, 'sigs':1}
    else:
        #To get best possible SNR, add the averages together (if it hasn't already been done):
        if not indat.flags['averaged']:
            av=op_averaging(indat)
        else:
            av=indat
        # also, for best results, we will combine all subspectra:
        if coilcombos is None:
            if indat.flags['isFourSteps']:
                av=op_fourStepCombine(av)
            if indat.dims['subSpecs']>-1:
                av=op_combinesubspecs(av,'summ')
        avfids=av.fids
        # Find the relative phases between the channels and populate the ph matrix
        # Won't avfids have different size depending on the size of fids??
        if coilcombos is None:
            # Jamie unwraps the phase in the Matlab version but only ever uses the angle. Maybe for viewing? Leaving it wrapped for now
            phs=[np.angle(avfids[phasept,nct,0])*180/np.pi for nct in range(avfids.shape[1])]
            if mode=='w':
                sigs=np.abs(avfids[phasept,:,0])
            elif mode=='h':
                S=np.max(np.abs(avfids[:,:,0]),axis=0)
                N=np.std(avfids[-100:,:,0],axis=0)
                sigs=S/(N**2)
        else:
            phs=coilcombos['phs']
            sigs=coilcombos['sigs']
        #now replicate the phase matrix to equal the size of the original matrix:
        sigs=sigs/np.linalg.norm(sigs.flatten())
        ph=np.ones(indat.sz)
        sig=np.ones(indat.sz)
        # and then adjust the phase in each dimension
        if indat.dims['coils']==0:
            for nct in range(indat.sz[0]):
                ph[nct,:]=phs[nct]*ph[nct,:]
                sig[nct,:]=sigs[nct]*sig[nct,:]
        elif indat.dims['coils']==1:
            for nct in range(indat.sz[1]):
                ph[:,nct,:]=phs[nct]*ph[:,nct,:]
                sig[:,nct,:]=sigs[nct]*sig[:,nct,:]
        elif indat.dims['coils']==2:
            for nct in range(indat.sz[2]):
                ph[:,:,nct,:]=phs[nct]*ph[:,:,nct,:]
                sig[:,:,nct,:]=sigs[nct]*sig[:,:,nct,:]
        elif indat.dims['coils']==3:
            for nct in range(indat.sz[3]):
                ph[:,:,:,nct,:]=phs[nct]*ph[:,:,:,nct,:]
                sig[:,:,:,nct,:]=sigs[nct]*sig[:,:,:,nct,:]
        elif indat.dims['coils']==4:
            for nct in range(indat.sz[4]):
                ph[:,:,:,:,nct,:]=phs[nct]*ph[:,:,:,:,nct,:]
                sig[:,:,:,:,nct,:]=sigs[nct]*sig[:,:,:,:,nct,:]
        #now apply the phases by multiplying the data by exp(-i*ph);
        fids=indat.fids*np.exp(-1j*ph*np.pi/180)
        fids_presum=fids
        specs_presum=fids.specs
        # Apply the amplitude factors my multiplying by the amplitude
        if mode=='w' or mode=='h':
            fids=fids*sig
        #Make the coilcombos structure:
        coilcombos={'phs':phs,'sigs':sigs}
        #now sum along coils dimension
        fids=np.sum(fids,axis=indat.dims['coils'])
        fids=np.squeeze(fids)
        outdat=indat.copy()
        outdat.fids=fids
        # change the dims variables
        for dimnm in indat.dims.keys():
            if indat.dims[dimnm]>indat.dims['coils']:
                outdat.dims[dimnm]=outdat.dims[dimnm]-1
        outdat.dims['coils']=-1
        outdat.flags['addedrcvrs']=True
        outdat.flags['writtentostruct']=True      
    return outdat,fids_presum,specs_presum,coilcombos

def op_combinesubspecs(indat,tmp):
    pass
    
# replaced op_addscans with the __add__ method in the FID object

def op_align_scans(inref, in1, tmax=None,mode='fp'):
    out1=in1.copy()
    def freqShiftComplexNest(in2,f):
        t=np.r_[0:(len(in2))*in1.dwelltime:in1.dwelltime]
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
        t=np.r_[0:(len(in2))*in1.dwelltime:in1.dwelltime]
        fid1=in2.flatten()
        y=add_phase(fid1*np.exp(-1j*t.T*f*2*np.pi),p)
        y=np.r_[np.real(y),np.imag(y)]
        return y
    # Note: don't need separate functions for final fid calculation because you
    # can just use existing op_freqshift and op_addphase functions with the 
    # fitted parameters (or defaults in 'f' and 'p' modes, as appropriate)
    if not inref.flags['addedrcvrs'] or not in1.flags['addedrcvrs']:
        raise TypeError('ERROR: Only makes sense to do this after channels have been combined.')
    # Jamie's original code also has a check that neither inref nor in1 is averaged, which doesn't make sense to me
    initpars=[0]*len(mode)
    if tmax is None:
        tmax=inref.t[-1]
    # curve_fit needs an array of floats to fit to, so putting the real and imaginary components
    # into one longer real-valued vector for fitting
    baseref=np.r_[np.real(inref.fids[np.logical_and(inref.t>=0,inref.t<tmax)]),np.imag(inref.fids[np.logical_and(inref.t>=0,inref.t<tmax)])]
    if mode=='f':
        parsFit,pcov=curve_fit(freqShiftComplexNest,in1.fids[np.logical_and(in1.t>=0,in1.t<tmax)].squeeze(),baseref.squeeze(),initpars)
        frq=parsFit[0]; ph=0;
    elif mode=='p':
        parsFit,pcov=curve_fit(phaseShiftComplexNest,in1.fids[np.logical_and(in1.t>=0,in1.t<tmax)].squeeze(),baseref.squeeze(),initpars)
        ph=parsFit[0]; frq=0;
    elif mode=='fp' or mode=='pf':
        parsFit,pcov=curve_fit(freqPhaseShiftComplexNest,in1.fids[np.logical_and(in1.t>=0,in1.t<tmax)].squeeze(),baseref.squeeze(),initpars)
        frq=parsFit[0]; ph=parsFit[1];
    else:
        raise TypeError('ERROR: unrecognized mode. Please enter either "f", "p" or "fp"')
    out1.fids=op_addphase(op_freqshift(in1,frq),ph)
    return out1, ph, frq
    
def op_alignScans_fd(in1, inref, ppmmin, ppmmax, tmax=0.1, mode='fp'):
    if not inref.flags['addedrcvrs'] or not in1.flags['addedrcvrs']:
        raise TypeError('ERROR: Only makes sense to do this after channels have been combined. ABORTING!')
    if not inref.flags['averaged'] or not in1.flags['averaged']:
        raise TypeError('ERROR: Only makes sense to do this after you have combined averages using op_averaging. ABORTING!')
    if inref.flags['isFourSteps'] or in1.flags['isFourSteps']:
        raise TypeError('ERROR: Only makes sense to do this after you have performed op_fourStepCombine. ABORTING!')

    baserange=op_freqrange(inref,ppmmin,ppmmax)
    startrange=op_freqrange(in1,ppmmin,ppmmax)
    # Rather than repeat the freqShiftComplexNext etc function here, I think that I can just call op_align_scans with the partial spectral in the frequency range
    outtmp,ph0,frq=op_align_scans(baserange, startrange, tmax=tmax,mode=mode)
    # That gives the values for aligning those sections but then need to apply
    # to the full spectrum
    out1=op_addphase(op_freqshift(in1,frq),ph0)
    return out1, ph0, frq

def op_align_all_scans(indat,fmin=2.8,fmax=3.1,ref='first',mode='fp'):
    """

    Parameters
    ----------
    indat : FID object
        input fid object, where indat.fids contain all data to be aligned.
    fmin : float, optional
        minimum frequency for spectral alignment in ppm. The default is 2.8.
    fmax : float, optional
        maximum frequency for spectral alignment in ppm. The default is 3.1.
    ref : char, optional
        indicates what to align  scans to. Choices are 'first' (first input) or
        'avg' (average of inputs). The default is 'first'.
    mode : char optional
        what to align: 'f' is frequency align only; 'p' is phase align only;
        'fp' is frequency and phase align. The default is 'fp'.

    Returns
    -------
    outdat : TYPE
        DESCRIPTION.

    """
    # first I need to go have a look at op_alignScans. Not sure what the freqPhseShiftNext
    # parts are doing.
    outdat=indat.copy()
    if ref=='first':
        #how to set this up to pick up the averaging dimension? I think I did this before
        refscan=indat.specs[:,0]
    elif ref=='avg':
        refscan=np.mean(indat.specs,axis=indat.dims['averages'])
    specs=indat.specs.copy()
    phvec=np.zeros([specs.shape[indat.dims['averages']],])
    frqvec=np.zeros([specs.shape[indat.dims['averages']],])
    # I think actually that I need to write op_align_scans first and figure out how that works
    return outdat

def op_autophase(indat,ppmmin,ppmmax,ph=0):
    if not indat.flags['zeropadded']:
        in_zp=op_zeropad(indat,10)
    else:
        in_zp=indat.copy()
    in_zp=op_freqrange(in_zp,ppmmin,ppmmax)
    ppmindex=np.argmax(np.abs(in_zp.specs[:,0]))
    ph0=-1*np.angle(in_zp.specs[ppmindex,0])*180/np.pi
    phShft=ph+ph0
    outdat=op_addphase(indat,phShft)
    return outdat,phShft

def op_averaging(indat):
    if indat.flags['averaged'] or indat.averages<2:
        print('WARNING: No averages found. Returning input without modification!')
        outdat=indat
    elif indat.dims['averages']==-1:
        print('WARNING: No averages found. Returning input without modification!')
        outdat=indat
    elif indat.dims['averages']!=-1:
        outdat=indat.copy()
        # average spectrum along averages dimension (previously it was a sum and divide by shape, but why?)
        outdat.fids=np.mean(indat.fids,axis=indat.dims['averages']).squeeze()
        if outdat.fids.ndim==1:
            outdat.fids=np.expand_dims(outdat.fids,axis=1)
        # change dims variable and update flags
        for eachvar in ['t','coils','subSpecs','extras']:
            if indat.dims[eachvar]>indat.dims['averages']:
                outdat.dims[eachvar]=indat.dims[eachvar]-1
        outdat.dims['averages']=-1
        outdat.averages=1
        outdat.flags['averaged']=True
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
    specpart=fullspec[indvals,:]
    if np.mod(len(np.nonzero(indvals)),2)==0:
        outdat.fids=fft(fftshift(specpart,axes=indat.dims['t']),axis=indat.dims['t'])
    else:
        outdat.fids=fft(np.roll(fftshift(specpart,axes=indat.dims['t']),1,axis=indat.dims['t']),axis=indat.dims['t'])
    # this will redefine the ppm range and then need to recalculate the spectral width.
    # There is probably a cleverer way to link these together via property decorators and setters
    outdat._ppmmax=ppmmin; outdat._ppmmin=ppmmax
    outdat.spectralwidth=np.abs(ppmmax-ppmmin)*(outdat.txfreq/1e6)
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
        for eachvar in ['t','coils','subSpecs','extras']:
            if indat.dims[eachvar]>indat.dims['averages']:
                outdat.dims[eachvar]=indat.dims[eachvar]-1
        outdat.dims['averages']=-1
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

def op_zeropad(indat,zpfact):
    outdat=indat.copy()
    if indat.flags['zeropadded']:
        cflag=input('Warning: zero padding has already been performed. Continue? (y or n): ')
        if cflag.lower()=='y':
            # creatine peak for water-suppressed data
            outdat.fids=np.zeros([zpfact*indat.sz[0],indat.sz[1]])+1j*np.zeros([zpfact*indat.sz[0],indat.sz[1]])
            outdat.fids[:indat.sz[0],:]=indat.fids
    else:
        outdat.fids=np.zeros([zpfact*indat.sz[0],indat.sz[1]])+1j*np.zeros([zpfact*indat.sz[0],indat.sz[1]])
        outdat.fids[:indat.sz[0],:]=indat.fids
    # Note that t, dwelltime, _ppmmin, ppmrange, etc are all calculated from 
    # spectralwidth, which should be unchanged here, and len(specs), which will
    # be adjusted. So no need to recalculate.
    outdat.flags['zeropadded']=True
    return outdat    

if __name__ == '__main__':
    """
    for debugging
    """
    pass
    
