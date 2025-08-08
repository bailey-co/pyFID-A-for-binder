#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:13:32 2022


fidA_io.py
Colleen Bailey, Sunnybrook Research Institute
Based on the FID-A Matlab code by Jamie Near, McGill University 2014.


"""
import os
import numpy as np
from scipy.fft import fftshift, ifft, fft
from datetime import date
import matplotlib.pyplot as plt
# Should I just import selected functions?
import fidA_processing as fop
GAMMA_DICT=fop.GAMMA_DICT

def get_default_flag_dict():
    flag_dict={'getftshifted': False,
     'filtered': False,
     'zeropadded': False,
     'freqcorrected': False,
     'phasecorrected': False,
     'subtracted': False,
     'writtentotext': False,
     'downsampled': False,
     'avgNormalized': False,
     'isFourSteps': False,
     'leftshifted': False,
     'writtentostruct': False,
     'gotparams': False,
     'averaged': False,
     'addedrcvrs': False}
    return flag_dict

class FID(object):
    """
    class FID(object)
    A FID object holds the fid that has been read in from a numpy array, either
    individual averages, coil readings, subspecs, or combinations. This information
    for each dimension is held in the dims attribute. There are also flags to
    describe what processing has been done and some sequence parameters.
    The spectrum "specs" from the fid and several other properties are calculated
    from existing attributes
    These objects can be processed using the functions in fidA_processing.
    """
    def __init__(self,fids,raw_avgs,spectralwidth,txfreq,te,tr,sequence=None,subSpecs=None,rawSubspecs=None,pts_to_left_shift=0,flags=None,dims=None,hdr=None,hdr_ext=None,nucleus=['1H'],center_freq_ppm=4.65):
        """
        FID(fids,raw_avgs,spectralwidth,txfreq,te,tr,sequence=None,subSpecs=None,rawSubspecs=None,pts_to_left_shift=0,flags=None,dims=None))

        """
        self.fids=fids
        # Want to fix this so that dims and flags are mandatory and the work is being
        # done in nii_to_mrs??
        if dims:
            self.dims=dims
        else:
            self.dims={'t':0}
        self.spectralwidth=spectralwidth
        self.nii_mrs=dict()
        if hdr is None:
            self.nii_mrs['hdr']={'hdr_text':'no header file'}
        else:
            self.nii_mrs['hdr']=hdr
        if hdr_ext is None:
            self.nii_mrs['hdr_ext']={'hdr_ext_text':'no header extension'}
        else:
            self.nii_mrs['hdr_ext']=hdr_ext
        self.nucleus=nucleus
        self.txfreq=txfreq
        self._GAMMA=GAMMA_DICT[self.nucleus[0].upper()]*1e6
        self.te=te
        self.tr=tr
        self.sequence=sequence
        self.date=date.today()
        self.rawAverages=raw_avgs
        self.averages=raw_avgs
        self.subSpecs=subSpecs
        self.rawSubspecs=rawSubspecs
        self.pointsToLeftshift=pts_to_left_shift
        self.added_ph0=0 #this may need to be a vector for rawdata files with multiple averages). Or may not be needed at all?
        self.added_ph1=0
        self.center_freq_ppm=center_freq_ppm
        if flags:
            self.flags=flags.copy()
        else:
            self.flags=get_default_flag_dict()
            self.flags['writtentostruct']=True
            self.flags['gotparams']=True
            if self.dims['coils']==-1:
                self.flags['addedrcvrs']=True
            if self.dims['averages']==-1:
                self.flags['averaged']=True
            if self.dims['subSpecs']==-1:
                self.flags['isFourSteps']=True
            else:
                self.flags['isFourSteps']=(self.fids.shape[self.dims['subSpecs']]==4)
    @property
    def specs(self):
        return fftshift(ifft(self.fids,axis=self.dims['t']),axes=self.dims['t'])
    @specs.setter
    def specs(self,newspec):
        # Set the fids object to the Fourier transform of the input newspec.
        # Note that ppm, etc are not adjusted, so you could create issues if
        # newspec has different spectral width, etc.
        print("Setting fids based on input specs. Please ensure spectral width and resolution match initial spectrum. No check on this is run.")
        if np.mod(newspec.shape[self.dims['t']],2)==0:
            self.fids=fft(fftshift(newspec,axes=self.dims['t']),axis=self.dims['t'])
        else:
            # From Matlab: have to do a circshift when the length of fids is 
            # odd so you don't introduce a small frequency shift into fids
            self.fids=fft(np.roll(fftshift(newspec,axes=self.dims['t']),1,axis=self.dims['t']),axis=self.dims['t'])
    @property
    def GAMMA(self):
        return self._GAMMA
    @GAMMA.setter
    def GAMMA(self,newGAMMA):
        # I debated whether this needed a setter since the idea will be to read
        # in the nucleus and then use a dict since the gyromagnetic ratios are
        # all known constants, but seems better to have this flexibility in case
        # I miss something. Note GAMMA should be in Hz.
        self._GAMMA=newGAMMA
    @property
    def Bo(self):
        return self.txfreq/self.GAMMA
    @Bo.setter
    def Bo(self,newBo):
        self.txfreq=newBo*self.GAMMA
    @property
    def spectralwidthppm(self):
        return self.spectralwidth/(self.txfreq/1e6)
    @spectralwidthppm.setter
    def spectralwidthppm(self,new_sw_ppm):
        self.spectralwidth=new_sw_ppm*(self.txfreq/1e6)
    @property
    def _ppmmin(self):
        return self.center_freq_ppm+self.spectralwidthppm/2
    @property
    def _ppmmax(self):
        return self.center_freq_ppm-self.spectralwidthppm/2
    @property
    def ppm(self):
        # to do the equivalent of what I've done with "t", I want to define ppm
        # in terms of center frequency, spectralwidth and self.specs.shape[0] and
        # any attempt to set it would adjust these three things (or, well, just
        # the center frequency and spectralwidth I think)
        return np.linspace(self._ppmmin,self._ppmmax,self.specs.shape[self.dims['t']])
    @property
    def dwelltime(self):
        return 1/self.spectralwidth
    @dwelltime.setter
    def dwelltime(self,newdwell):
        self.spectralwidth=1/newdwell
    @property
    def t(self):
        #t2=np.r_[0:self.fids.shape[self.dims['t']]*self.dwelltime:self.dwelltime]
        t2=np.linspace(0,self.fids.shape[self.dims['t']]*self.dwelltime,self.fids.shape[self.dims['t']]+1)[:-1]
        # Corrects for potential rounding errors
        #if len(t2)!=self.fids.shape[self.dims['t']]:
        #    t2=np.r_[0:(self.fids.shape[self.dims['t']]-1)*self.dwelltime:self.dwelltime]
        return t2
    @property
    def sz(self):
        return self.specs.shape
    @property
    def ndim(self):
        return len(self.sz)
    def __mul__(self,mult1):
        if isinstance(mult1, FID):
            out1=self.copy()
            out1.fids=self.fids*mult1.fids
            return out1
        elif isinstance(mult1, int) or isinstance(mult1, float):
            out1=self.copy()
            out1.fids=self.fids*mult1
            return out1
        else:
            raise TypeError(f"sorry, don't know how to multiply by {type(mult1).__name__}")
    def __rmul__(self,mult1):
        if isinstance(mult1, int) or isinstance(mult1, float):
            out1=self.copy()
            out1.fids=self.fids*mult1
            return out1
        else:
            raise TypeError(f"sorry, don't know how to multiply by {type(mult1).__name__}")
    def __add__(self,add1):
            if isinstance(add1, FID):
                out1=self.copy()
                out1.fids=self.fids+add1.fids
                return out1
            elif isinstance(add1, int) or isinstance(add1, float):
                out1=self.copy()
                out1.fids=self.fids+add1
                return out1
            else:
                raise TypeError(f"sorry, don't know how to add {type(add1).__name__}")
    def __sub__(self,add1):
            if isinstance(add1, FID):
                out1=self.copy()
                out1.fids=self.fids-add1.fids
                return out1
            elif isinstance(add1, int) or isinstance(add1, float):
                out1=self.copy()
                out1.fids=self.fids-add1
                return out1
            else:
                raise TypeError(f"sorry, don't know how to add {type(add1).__name__}")
    def __div__(self,mult1):
        if isinstance(mult1, FID):
            out1=self.copy()
            out1.fids=self.fids/mult1.fids
            return out1
        elif isinstance(mult1, int) or isinstance(mult1, float):
            out1=self.copy()
            out1.fids=self.fids/mult1
            return out1
        else:
            raise TypeError(f"sorry, don't know how to multiply by {type(mult1).__name__}")
    def __truediv__(self,mult1):
        return self.__div__(mult1)
    def __repr__(self):
        dimlist=list()
        for kct in range(self.ndim):
            dimlist.append(self.get_dimnm_from_idx(kct))
        return '{:s} has fids size {:s} and dimensions {:s}'.format(type(self).__name__,str(self.sz),str(dimlist))
    def __contains__(self,item):
        if item in self.dims:
            if self.dims[item]>-1:
                return True
        return False
    def __getitem__(self, key):
        outdat=self.copy()
        outdat.fids=outdat.fids[key]
        for keyct,eachdim in enumerate(key):
            if type(eachdim) is int:
                # find the dims variable with that dimension
                dimnm=outdat.get_dimnm_from_idx(keyct)#[dimnm for dimnm,dimidx in self.dims.items() if dimidx==keyct][0]
                # change this dimval to -1 and decrease all dimensions above
                outdat._remove_dim_from_dict(dimnm)
        return outdat
    def __setitem__(self, key, newfid):
        # Note: does not currently allow number of dimensions to change.
        if self.ndim != newfid.ndim:
            raise Exception('ERROR: Assigning to slice of FID object cannot alter number of dimensions. \n self.ndim={:d}, newfid.ndim={:d}'.format(self.ndim,newfid.ndim))
        self.fids[key]=newfid
    
    def _remove_dim_from_dict(self,key):
        # I figure that this should be a private method, since users should not
        # generally be removing dimensions from the dictionary because they need
        # to match the dimensions of self.fids. However, there are many fidA_processsing
        # functions that involve removing a dimension (eg. op_addrcvrs removes 
        # the 'coil' dimension), so I've made it single underscore. Please code 
        # responsibly.
        # Set the indicated dimension from self.dims to -1 and reduce the index
        # for each dimension above it by 1. No return value
        for dimnm in self.dims.keys():
            if self.dims[dimnm]==self.dims[key]:
                self.dims[dimnm]=-1
            elif self.dims[dimnm]>self.dims[key]:
                self.dims[dimnm]=self.dims[dimnm]-1
                
    def get_dimnm_from_idx(self, idx):
        keynm_list=[kn for kn,dval in self.dims.items() if dval==idx]
        if len(keynm_list)==0:
            print('No dimenions with index {:d} in dims. Returning value None'.format(idx))
            return None
        elif len(keynm_list)==1:
            return keynm_list[0]
        else:
            raise Exception('ERROR: More than one key with dimension {:d}'.format(idx))
        
    def copy(self):
        newfid=FID(self.fids.copy(),self.rawAverages,self.spectralwidth,self.txfreq,
                   self.te,self.tr,self.sequence,self.subSpecs,self.rawSubspecs,
                   self.pointsToLeftshift,self.flags.copy(),self.dims.copy(),
                   self.nii_mrs['hdr'].copy(),self.nii_mrs['hdr_ext'].copy(),
                   self.nucleus,self.center_freq_ppm)
        newfid.date=date.today()
        newfid.averages=self.averages
        return newfid
    def plot_spec(self,xlims=[4.5,0],xlab='Chemical Shift (ppm)',ylab='Signal',title='',plotax=None, **kwargs):
        # Need to update to deal with multiple averages and other possible dimensions, as in Matlab op_plotspec
        if plotax is None:
            [f1,plotax]=plt.subplots(1,1)
        # This attempt to generate a plot for multiple coils is bad. Whatever is happening with the
        # Fourier Transform seems to be adding in phasing or something. The absolute spectra
        # look okay but the real component has lots of weird phase additions (although maybe that
        # is to be expected and Bruker deals with that automatically in processing)
        # plt.plot(np.abs(fop.op_averaging(indat).specs)) looks very different than indat.plot_specs()
        if self.fids.ndim==3: #probably a more generic way to do this, but for Bruker multi-coil, I'll combine averages to see each coil
            #spec_for_plot=np.real(np.mean(self.specs,axis=self.dims['averages']))
            spec_for_plot=np.real(fop.op_averaging(self).specs)
        else:
            spec_for_plot=np.real(self.specs)
        plotax.plot(self.ppm,spec_for_plot,**kwargs)
        plotax.set_xlim(xlims)
        plotax.set_xlabel(xlab)
        plotax.set_ylabel(ylab)
        plotax.set_title(title)

def fid_from_specs(oldspec):
    # Should this be here or in processing? Currently only called in io_bruker_load
    # when loading 2dseq but may be needed for other vendors and formats??
    # Need to recalculate fid from spec, but you have to do a circshift when the
    # length of fids is odd so you don't introduce a small frequency shift into fids
    # Function assumes that time is the first dimension.
    if np.mod(oldspec.shape[0],2)==0:
        newfids=fft(fftshift(oldspec,axes=0),axis=0)
    else:
        newfids=fft(np.roll(fftshift(oldspec,axes=0),1,axis=0),axis=0)
    return newfids
    
def get_par(fname,parname,vartype='float'):
    """
    raw_par=get_par(fname,parname,vartype='float')
    Helper file to return the value for a particular parameter in a Bruker
    method, proc, etc. file

    Parameters
    ----------
    fname : FILE STRING
        Full or relative path to file to open.
    parname : STRING
        Variable name to find, including '$' but not '='. eg. $PVM_DigNp.
    vartype : STRING, optional
        Variable type, used to convert from string before return. The default is 'float'.

    Returns
    -------
    raw_par : float, int or string, depending on value of 'vartype'
        Value of parname.
    """
    line=''
    with open(fname) as f:
        while parname+'=' not in line:
            line=f.readline()
    if vartype=='float':
        raw_par=float(line[line.find('=')+1:])
    elif vartype=='int':
        raw_par=int(line[line.find('=')+1:])
    else:
        raw_par=line[line.find('=')+1:].strip()
    return raw_par
        
def io_loadspec_GE(fname,subspecs=1):
    # Using the existing spec2nii to load into nifti_mrs and then convert that 
    # into the FID object
    # Right now, the fn_out filename is unused from this call.
    data,fnames=read_pfile(fname,'fn_out.txt')
    # note that data is a list of len(2) in this case.
    # It looks like the data are in data[0].image.data and dimensions are 
    # [cols,rows,slices,numTimePts, numCoils,numSpecPts] where row,cols,slice are all 1 for SVS data
    out1=data[0]
    hdr=out1.header
    hdr_ext=out1.hdr_ext.to_dict()
    fids=out1.image.data
    f0=hdr_ext['SpectrometerFrequency']
    dt=out1.dwelltime
    sw=1/dt
    # These are the initial dimensions for NIfTI MRS. Will permute later for FID-A
    dims=dict()
    dims['x']=0; dims['y']=1; dims['z']=2; dims['t']=3
    fidA_dictnames=['coils','averages','subSpecs','subSpecs','extras','extras','extras','extras','extras','extras','extras','extras']
    nift_dictnames=['DIM_COIL','DIM_DYN','DIM_EDIT','DIM_ISIS','DIM_INDIRECT_0','DIM_INDIRECT_1','DIM_INDIRECT_2','DIM_PHASE_CYCLE','DIM_MEAS','DIM_USER_0','DIM_USER_1','DIM_USER_2']
    for dctnm, niftinm in zip(fidA_dictnames,nift_dictnames):
        try:
            dims[dctnm]=out1.dim_tags.index(niftinm)+4
        # We set any missing dimensions to -1
        except ValueError:
            if dctnm not in dims.keys(): # some key names are repeated because NIFTI has more possible dimensions than fidA and we don't want to erase things already set
                dims[dctnm]=-1
    allDims=out1.shape #FidA io_loadspec_niimrs excludes the first dimension for some reason
    # Find the number of averages. 'averages' will specify the current number of averages in 
    # the dataset as it is proposed, which may be subject to change. 'rawAverages' will specify
    # the original number of acquired averages in the dataset, which is unchangeable.
    if dims['subSpecs']!=-1:
        if dims['averages']!=-1:
            averages=allDims[dims['averages']]*allDims[dims['subSpecs']]
            rawAverages=averages
        else:
            averages=allDims[dims['subSpecs']]
            rawAverages=1
    else:
        if dims['averages']!=-1:
            averages=allDims[dims['averages']]
            rawAverages=averages
        else:
            averages=1
            rawAverages=1
    # Find the number of subspecs
    # 'subSpecs' will specify the current number of subspectra and 'rawSubspecs'
    # will specify the original number of acquired subspectra in the dataset
    if dims['subSpecs']!=-1:
        subSpecs=allDims[dims['subSpecs']]
        rawSubspecs=subSpecs
    else:
        subSpecs=1
        rawSubspecs=subSpecs
    # Order the data and dimensions
    if allDims[0]*allDims[1]*allDims[2]==1: #SVS, but there is no other case currently written in Matlab
        dims['x']=-1
        dims['y']=-1
        dims['z']=-1
        fids=np.squeeze(fids)
        # permute so the order is [time domain,coils,averages,subSpecs,extras]
        dims={dimname:dimval-3 if dimval!=-1 else dimval for dimname,dimval in dims.items()}
        sqzDims=[dimname for dimname,dimval in dims.items() if dimval!=-1]
        # The Matlab code for io_loadspec_niimrs is quite long and repetitive, 
        # but I think that it should be possible to just reorder sqzDims since 
        # all possible dimensions are in dims.keys() but just -1 if not present 
        # (even if they weren't) present, adding a try, catch KeyError should work.
        dimorder=['t','coils','averages','subSpecs','extras']
        sqzDims=[dimname for dimname in dimorder if dims[dimname]!=-1]
        fids=np.transpose(fids,[dims[kval] for kval in sqzDims])
        # then we need to reassign the values in dims to reflect the new ordering
        for dimct,dimnm in enumerate(sqzDims):
            dims[dimnm]=dimct
        # Compared to NIfTI MRS, fidA apparently needs the conjugate
        fids=np.conj(fids)
        flagpars=['getftshifted','filtered','zeropadded','freqcorrected','phasecorrected',
                  'subtracted','writtentotext','downsampled','avgNormalized',
                  'isFourSteps','leftshifted']
        flagdict={parnm:False for parnm in flagpars}
        flagdict['writtentostruct']=True
        flagdict['gotparams']=True
        if dims['averages']==-1:
            flagdict['averages']=True
        else:
            flagdict['averages']=False
        if dims['coils']==-1:
            flagdict['addedrcvrs']=True
        else:
            flagdict['addedrcvrs']=False
        if dims['subSpecs']==-1:
            flagdict['isFourSteps']=False
        else:
            flagdict['isFourSteps']=(fids.shape[dims['subSpecs']]==4)
        # Are there cases where the out1.spectrometer_frequency list will be longer?
        outnii=FID(fids,rawAverages,sw,out1.spectrometer_frequency[0]*1e6,hdr_ext['EchoTime']*1000,hdr_ext['RepetitionTime']*1000,sequence=hdr_ext['SequenceName'],subSpecs=subSpecs,rawSubspecs=rawSubspecs,dims=dims,flags=flagdict)
        # Matlab's load for niimrs saves some extra stuff, like the nucleus and 
        # hdr. Need to put these in the __init__ for the FID object if I want to
        # have them. Not currently needed for fidA, where some functions use 
        # GAMMAP and so assume proton, but this would be easy enough to change.
        #outnii.nucleus=out1.nucleus
        #outnii.nii_mrs.hdr=hdr
        #outnii.nii_mrs.hdr_ext=hdr_ext
    # Note that I probably need to do the same thing for data[1]. I assume one of these is the reference.
    # Also, I intended to write an niimrs to fidA function but some things are probably
    # vendor-specific so I'll have to work out which those are.
    # I think that this could be shortened up a lot by recognizing what is already being done in the FIDS object
    # Also, I have the subspec argument at the top that I need to see if I need by checking against io_loadspec_GE.m
    return outnii

def io_loadspec_bruk(inDir,spectrometer=False,try_raw=False,info_dict=False,ADC_OFFSET=68):
    # Get relevant parameters (this could be done more efficiently by opening files once at the start and reading all relevant parameters from one opening)
    if spectrometer:
        dic1=read_jcamp_spec(os.path.join(inDir,'acqu'))
        spectralwidth=dic1['SW_h']
        txfrq=dic1['SFO1']*1e6
        te=-1
        tr=-1
        sequence=dic1['PULPROG'][1:-1]
        if info_dict:
            info_dict=dic1
        else:
            info_dict=None
    else:
        spectralwidth=get_par(os.path.join(inDir,'method'),'$PVM_DigSw')
        txfrq=get_par(os.path.join(inDir,'acqp'),'$BF1')*1e6
        te=get_par(os.path.join(inDir,'method'),'$PVM_EchoTime')
        tr=get_par(os.path.join(inDir,'method'),'$PVM_RepetitionTime')
        sequence=get_par(os.path.join(inDir,'method'),'$Method','string')
        if info_dict:
            info_dict=read_jcamp_spec(os.path.join(inDir,'method'))
        else:
            info_dict=None
    # Specify the number of subspecs.  For now, this will always be one.
    subSpecs=1; rawSubspecs=1
    
    def get_spec(fname,raw_avgs=None):
        fid_data=np.fromfile(fname,dtype=np.int32)
        real_fid = fid_data[::2]
        imag_fid = fid_data[1::2]
        fids_raw=real_fid+1j*imag_fid
        if fname.endswith('fid.raw') or fname.endswith('rawdata.job0'):
            # I need more info here. When you use the 2x2 linear array, the rawdata
            # file includes info from all 4 coils, so I should try to read that
            # in and split it up. Not sure if there is a number of coils parameter
            # but you could read in averages, number of points and then everything else
            raw_avgs=get_par(os.path.join(inDir,'method'),'$PVM_NAverages','int')
            expmode=get_par(os.path.join(inDir,'acqp'),'$ACQ_experiment_mode','string')
            if expmode.strip()=='ParallelExperiment':
                ncoil=int(get_par(os.path.join(inDir,'acqp'),'$ACQ_ReceiverSelect','string').split()[1])
                coildim=1
                avgdim=2
                fids_raw=np.reshape(fids_raw,[raw_avgs*ncoil,-1]).T
            else:
                coildim=-1
                avgdim=1
                fids_raw=np.reshape(fids_raw,[raw_avgs,-1]).T
            # So now the problem is that I need to reshape for the number of coils
            # But this will affect the number of dimensions, which in turn affects
            # the fid_trunc calculation and maybe the pad??
        elif fname.endswith('fid'):
            if os.path.exists(os.path.join(inDir,'method')):
                raw_dat_pts=get_par(os.path.join(inDir,'method'),'$PVM_DigNp','int')
            else:
                raw_dat_pts=len(real_fid)
            raw_avgs=int(np.shape(real_fid)[0]/raw_dat_pts)
            if np.mod(real_fid.shape[0],raw_dat_pts)!=0:
                print('number of repetitions cannot be accurately found')
            fids_raw=np.reshape(fids_raw,[-1,raw_dat_pts]).T
            avgdim=-1
            coildim=-1
        elif fname.endswith('fid.ref'):
            fids_raw=np.reshape(fids_raw,[raw_avgs,-1]).T
            avgdim=1
            coildim=-1
        elif fname.endswith('fid.refscan'):
            raw_dat_pts=len(real_fid)
            raw_avgs=1
            if np.mod(real_fid.shape[0],raw_dat_pts)!=0:
                print('number of repetitions cannot be accurately found for refscan file')
            fids_raw=np.reshape(fids_raw,[-1,raw_dat_pts]).T
            avgdim=-1
            coildim=-1
        try:
            fids_trunc=fids_raw[ADC_OFFSET:,:]
        except IndexError:
            fids_trunc=np.expand_dims(fids_raw,axis=1)
            fids_trunc=fids_trunc[ADC_OFFSET:,:]
        fids=np.pad(fids_trunc, pad_width=[[0,ADC_OFFSET],[0,0]])
        if coildim!=-1:
            fids=np.transpose(np.reshape(fids,[-1,raw_avgs,ncoil]),[0,2,1])
            #fids=np.reshape(fids,[-1,ncoil,raw_avgs])
        # Broken for this. Hard-coding the dimensions.
        tmpdim={'t': 0, 'averages': 2, 'subSpecs': -1, 'coils': 1, 'extras': -1}
        fid1=FID(fids,raw_avgs,spectralwidth,txfrq,te,tr,sequence,subSpecs,rawSubspecs,dims=tmpdim)
        fid1.dims['averages']=avgdim
        fid1.dims['t']=0
        fid1.dims['coils']=coildim
        fid1.dims['subSpecs']=-1
        fid1.dims['extras']=-1
        if fid1.dims['subSpecs']==0:
            fid1.flags['isFourSteps']=0
        else:
            fid1.flags['isFourSteps']=(fid1.sz[fid1.dims['subSpecs']]==4)
        return fid1,raw_avgs
    
    # First try to load fid.raw file. If that does not work, use regular fid
    if try_raw:
        try:
            outfid,rawavg1=get_spec(os.path.join(inDir,'rawdata.job0'))
        except FileNotFoundError:
            outfid,rawavg1=get_spec(os.path.join(inDir,'fid.raw'))
    else:
        #untested since I don't have an example dataset with .raw missing
        print('WARNING: /fid.raw not found. Using /fid ....')
        outfid,rawavg1=get_spec(os.path.join(inDir,'fid'))
        
    # NOW TRY LOADING IN THE REFERENCE SCAN DATA (IF IT EXISTS)
    # In PV6.0.1, I think the reference scan data is just saved in fid.refscan
    # We never do more than one average, so I'm not sure if averages are 
    # ever saved separately. In any case, rawdata.job1 is the navigator so
    # that's not the reference scan raw data and I've removed the if try_raw block
    if os.path.isfile(os.path.join(inDir,'fid.ref')):
        reffid,rawavg2=get_spec(os.path.join(inDir,'fid.ref'),raw_avgs=rawavg1)
    elif os.path.isfile(os.path.join(inDir,'fid.refscan')):
        reffid,rawavg2=get_spec(os.path.join(inDir,'fid.refscan'),raw_avgs=rawavg1)
    else:
        print('Could not find reference scan. Skipping')
        reffid=0
    return outfid,reffid,info_dict

def read_jcamp_spec(fname):
    """
    read_jcamp_spec(fname)
    Opens a Bruker file with scan information and reads in each variable to a dictionary
    
    inputs:
    fname - string of the Bruker file
                
    returns:
    dic - a dictionary of scan parameters from the file fname. It attempts to get the datatype of the values correct
          (int, float, string; single values or arrays (which are saved in Python's list format)). Some arrays 
          (eg. ACQ_coil_elements) are arrays of lists in brackets and these are left as a single string rather than
          converting into a list of lists.
    """    
    dic=dict()

    #read everything that's not a comment ("$$) or header("##(not$)")
    with open(fname,'r') as f:
        linelist=[line.rstrip() for line in f if (line[:2]!="$$" and not (line[:2]=="##" and line[2]!="$"))]
                
    #The file has line breaks for some parameters, some of which are arrays and some of which are not. The strategy 
    #I've taken is to join the entire file into one line, separated by spaces, then split again wherever there's a 
    #comment indicator (##$). This should put each parameter on its own line. The first element of the new list 
    #will be empty since the file starts with ##$
    plist=" ".join(linelist).split("##$")[1:]
    for par in plist:
        dkey,dval=par.split('=',maxsplit=1)        
        if "(" in dval and " )" in dval: #array case
            try:
                arrShape=[int(d.strip()) for d in dval[dval.index("(")+2:dval.index(" )")].split(', ')]
                dval=dval[dval.index(" )")+3:] #everything after the array size is the variable we want
                olddval=dval #save in case the array reshape doesn't work (which should mean a single value string)
                dval=[type_it(delem.strip()) for delem in dval.split()]
                try: #sometimes it just gives you the shape with no values
                    np.array(dval).reshape(arrShape)
                    dic[dkey]=dval
                except ValueError:
                    dic[dkey]=olddval #if we can't get the shape right, just leave it
            except ValueError: #for CPMG sequence, one of the excitatory pulses has a space before ')' and gets read as an array.
                dval=dval[dval.index("(")+1:dval.index(")")].split(", ")
                dval=[type_it(delem.strip()) for delem in dval]
                dic[dkey]=dval        
        elif "(" in dval and " )" not in dval: #multi-line parameter case
            #here there is no array size, so the thing in the brackets is the parameter we want
            dval=dval[dval.index("(")+1:dval.index(")")].split(", ")
            dval=[type_it(delem.strip()) for delem in dval]
            dic[dkey]=dval
        else: #simple scalar parameter
            dic[dkey]=type_it(dval.strip())     
    return dic
        
def type_it(dval):
    """
    type_it(dval)
    Values read in using fread are strings. This should cast them as the correct type
    
    inputs:
    dval - string of the parameter value
    returns:
    dval - parameter value recast as the correct type (float, int or string)
    """
    try:
        dval=int(dval)
    except ValueError:
        try:
            dval=float(dval)
        except ValueError:
            pass #string case
    return dval

def io_readlcmcoord(fname):
    lcm_dict=dict()
    info_dict=dict()
    with open(fname) as f:
        coord_text=f.read()
    div_strs=[' points on ppm-axis = NY','NY phased data points follow',
              'NY points of the fit to the data follow','NY background values follow',
              ' Conc. = ']
    coord_lines=coord_text.split('\n')
    nlines=int(coord_lines[2].split()[0])
    info_dict['names']=[eachlines.split()[3] for eachlines in coord_lines[4:4+nlines-1]]
    info_dict['conc']=[float(eachlines.split()[0]) for eachlines in coord_lines[4:4+nlines-1]]
    info_dict['SD_perc']=[int(eachlines.split()[1][:-1]) for eachlines in coord_lines[4:4+nlines-1]]
    info_dict['SNR']=int(coord_lines[4+nlines].split()[-1])
    tmpline=coord_lines[4+nlines]
    info_dict['FWHM']=float(tmpline[tmpline.index('=')+1:tmpline.index('ppm')])
    tmpline=coord_lines[4+nlines+1]
    info_dict['shift']=float(tmpline[tmpline.index('=')+1:tmpline.index('ppm')])
    tmpline=coord_lines[4+nlines+2]
    info_dict['phase']=float(tmpline[tmpline.index('deg')+3:tmpline.index('deg/ppm')])
    for divct,(divstr,partstr) in enumerate(zip(div_strs[:-1],['ppm','data','fit','bgrd'])):
        tmppts=coord_text[coord_text.find(divstr)+len(divstr):coord_text.find(div_strs[divct+1])]
        # one possibility is to save each of these as a FID object so that you can use the same
        # functions and stuff that you do for raw data but not sure how useful this is. Mostly
        # it might work for plotting but also you often want to plot multiple metabolites as a 
        # ridgeplot and I already have a function for that.
        if partstr=='bgrd': # Need to cut metabolite name after split. Can't specify metabolite in string because depends on basis set
            lcm_dict[partstr]=np.array([float(numstr) for numstr in tmppts.split()[:-1]])
        else:
            lcm_dict[partstr]=np.array([float(numstr) for numstr in tmppts.split()])
    # split up text into chunks for each metabolite
    tmpconc=coord_text.split('Conc. = ')
    npts=len(lcm_dict['ppm'])+1
    for concct,eachconc in enumerate(tmpconc[1:]):
        metname=tmpconc[concct].split()
        metname=metname[-1].strip()
        #print(metname)
        lcm_dict[metname]=np.array([float(numstr) for numstr in eachconc.split()[1:npts]])
    return lcm_dict, info_dict

    
def io_writelcm(infid,outfile,te=None,vol=8.0):
    # Need to add the error checks, equivalent to those in Matlab
    RF=np.zeros([infid.fids.shape[infid.dims['t']],2])
    RF[:,0]=np.imag(infid.fids[:,0])
    RF[:,1]=np.real(infid.fids[:,0])
    with open(outfile,'w') as f:
        f.write(' $SEQPAR')
        f.write('\n echot= {:4.3f}'.format(infid.te))
        f.write("\n seq= 'PRESS'")
        f.write('\n hzpppm= {:5.6f}'.format(infid.txfreq/1e6))
        f.write('\n NumberOfPoints= {:n}'.format(infid.fids.shape[infid.dims['t']]))
        f.write('\n dwellTime= {:5.6f}'.format(infid.dwelltime))
        f.write('\n $END')
        f.write('\n $NMID')
        f.write("\n id='ANONYMOUS ', fmtdat='(2E15.6)'")
        f.write('\n volume={:4.3e}'.format(vol))
        f.write('\n tramp=1.0')
        f.write('\n $END\n')
        for eachct in range(RF.shape[0]):
            f.write('  {:7.6e}  {:7.6e}\n'.format(RF[eachct,0],RF[eachct,1]))

if __name__ == '__main__':
    """
    for debugging
    """
    fname='data/6'
    out1,ref1,infodict=io_loadspec_bruk(fname,try_raw=True)
    print(out1.sz)
