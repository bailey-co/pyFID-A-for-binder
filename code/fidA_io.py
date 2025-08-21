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
from pyFidA.code.fidA_processing import op_averaging
from pyFidA.code.fidA_processing import GAMMA_DICT
from spec2nii.GE.ge_pfile import read_pfile
import pkg_resources
import sys

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
        self.nii_mrs['hdr']=hdr
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
        #print("Setting fids based on input specs. Please ensure spectral width and resolution match initial spectrum. No check on this is run.")
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
        # in the nucleus and then use a dict of gyromagnetic ratios, which are 
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
        # in terms of center frequency, spectralwidth and self.specs.shape[0].
        # Not sure if a setter could be done. Would need to make assumptions 
        # about what changed
        return np.linspace(self._ppmmin,self._ppmmax,self.specs.shape[self.dims['t']])
    @property
    def dwelltime(self):
        return 1/self.spectralwidth
    @dwelltime.setter
    def dwelltime(self,newdwell):
        self.spectralwidth=1/newdwell
    @property
    def t(self):
        t2=np.linspace(0,self.fids.shape[self.dims['t']]*self.dwelltime,self.fids.shape[self.dims['t']]+1)[:-1]
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
        # Note that you have to go through the dimensions in reverse order so that
        # you aren't changing dimension numbers that you want to reference later
        for keyct,eachdim in enumerate(key[::-1]):
            if type(eachdim) is int:
                # find the dims variable with that dimension
                dimnm=outdat.get_dimnm_from_idx(len(key)-1-keyct)#
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
        if self.nii_mrs['hdr'] is None:
            new_nii_mrs_hdr=None
        else:
            new_nii_mrs_hdr=self.nii_mrs['hdr'].copy()
        if self.nii_mrs['hdr_ext'] is None:
            new_nii_mrs_hdr_ext=None
        else:
            new_nii_mrs_hdr_ext=self.nii_mrs['hdr_ext'].copy()
        newfid=FID(self.fids.copy(),self.rawAverages,self.spectralwidth,self.txfreq,
                   self.te,self.tr,self.sequence,self.subSpecs,self.rawSubspecs,
                   self.pointsToLeftshift,self.flags.copy(),self.dims.copy(),
                   new_nii_mrs_hdr,new_nii_mrs_hdr_ext,
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
            spec_for_plot=np.real(op_averaging(self).specs)
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

def io_loadspec_GE(fname,subspecs=1):
    # Using the existing spec2nii to load into nifti_mrs and then convert that 
    # into the FID object
    # Right now, the fn_out filename is unused from this call.
    data,fnames=read_pfile(fname,'fn_out.txt')
    # note that data is a list of len(2) in this case. For the moment, I've just
    # got the first element being sent to the nii_to_fidA converter as a test.
    # Eventually, I'll want to send and convert both to return both the unsuppressed
    # and water-suppressed reference data (not even sure which one is which)
    outnii=nii_to_fidA(data[0])
    return outnii
    
def nii_to_fidA(out1):
    # It looks like the data are in out1.image.data and dimensions are 
    # [cols,rows,slices,numTimePts, numCoils,numSpecPts] where row,cols,slice are all 1 for SVS data
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
        # In fidA in Matlab, there are always at least 2 dimensions, even if
        # one of them has size 1. I was using expand_dims to mimic this, but it
        # makes some processing functions more awkward, so on 20250509, I removed
        # this and now you can have 't' as the only dimensions.
        #if len(fids.shape)==1:
        #    fids=np.expand_dims(fids, axis=1)
        #    dims['averages']=4
        # permute so the order is [time domain,coils,averages,subSpecs,extras]
        dims={dimname:dimval-3 if (dimval!=-1) else -1 for dimname,dimval in dims.items()}
        # The Matlab code for io_loadspec_niimrs is quite long and repetitive, 
        # but I think that it should be possible to just reorder sqzDims since 
        # all possible dimensions are in dims.keys() but just -1 if not present 
        # (even if they weren't) present, adding a try, catch KeyError should work.
        dimorder=['t','coils','averages','subSpecs','extras']
        sqzDims=[dimname for dimname in dimorder if (dims[dimname]!=-1 and dims[dimname]<len(fids.shape))]
        fids=np.transpose(fids,[dims[kval] for kval in sqzDims])
        # then we need to reassign the values in dims to reflect the new ordering
        for dimct,dimnm in enumerate(sqzDims):
            dims[dimnm]=dimct
        # Compared to NIfTI MRS, fidA apparently needs the conjugate
        fids=np.conj(fids)
        flagdict=get_default_flag_dict()
        flagdict['writtentostruct']=True
        flagdict['gotparams']=True
        # This should fix the issues of checking whether data that are imported
        # without, eg. multiple coils are interpreted as receivers being addded.
        # Another thought would be to replace some of the flags with properties
        # so that they just check whether that dimension is -1 (and I don't allow
        # dimensions with size 1)
        if dims['averages']==-1:
            flagdict['averaged']=True
        else:
            flagdict['averaged']=False
        if dims['coils']==-1:
            flagdict['addedrcvrs']=True
        else:
            flagdict['addedrcvrs']=False
        if dims['subSpecs']==-1:
            flagdict['isFourSteps']=False
        else:
            flagdict['isFourSteps']=(fids.shape[dims['subSpecs']]==4)
        # Are there cases where the out1.spectrometer_frequency list will be longer?
        outnii=FID(fids,rawAverages,sw,out1.spectrometer_frequency[0]*1e6,hdr_ext['EchoTime']*1000,hdr_ext['RepetitionTime']*1000,sequence=hdr_ext['SequenceName'],subSpecs=subSpecs,rawSubspecs=rawSubspecs,dims=dims,flags=flagdict,hdr=hdr,hdr_ext=hdr_ext,nucleus=out1.nucleus)
        # Matlab's load for niimrs saves some extra stuff, like the nucleus and 
        # hdr. I've added these to the FID.__init__ with default None for other
        # loading functions. Not currently needed for fidA, where some functions use 
        # GAMMAP and so assume proton, but this would be easy enough to change.
        #outnii.nucleus=out1.nucleus
        #outnii.nii_mrs.hdr=hdr
        #outnii.nii_mrs.hdr_ext=hdr_ext
    # I think that this could be shortened up a lot by recognizing what is already being done in the FIDS object
    # Also, I have the subspec argument at the top that I need to see if I need by checking against io_loadspec_GE.m
    return outnii
    
def io_writelcm(infid,outfile,te=None,vol=8.0):
    # Not including any of the error checks at this point. See Jamie's Matlab file
    RF=np.zeros([infid.fids.shape[infid.dims['t']],2])
    # only works for 1D. Also, there are differences between vendors here that I haven't correctly
    # dealt with. Only really done with Bruker PV6.0.1
    RF[:,1]=np.imag(infid.fids)
    RF[:,0]=np.real(infid.fids)
    with open(outfile,'w') as f:
        f.write(' $SEQPAR')
        if te is None:
            f.write('\n echot= {:4.3f}'.format(infid.te))
        else:
            f.write('\n echot= {:4.3f}'.format(te))
        f.write("\n seq= 'PRESS'")
        f.write('\n hzpppm= {:5.6f}'.format(infid.txfreq/1e6))
        f.write('\n NumberOfPoints= {:n}'.format(infid.fids.shape[infid.dims['t']]))
        f.write('\n dwellTime= {:5.6f}'.format(infid.dwelltime))
        f.write('\n $END')
        f.write('\n $NMID')
        f.write('\n bruker=T')
        f.write('\n seqacq=F')
        f.write("\n id='ANONYMOUS ', fmtdat='(2E14.5)'")
        #f.write("\n fmtdat='(2E14.5)'")
        f.write('\n volume={:4.3e}'.format(vol))
        f.write('\n tramp=1.0')
        f.write('\n $END\n')
        for eachct in range(RF.shape[0]):
            f.write('  {:7.5e}  {:7.5e}\n'.format(RF[eachct,0],RF[eachct,1]))
            
def io_loadspec_bruk(fname, load_ref=False, do_leftshift=True, fill_truncated_data=True):
    """
    A lot of this is similar to what is done in bruker.py in the read_bruker() 
    function, but that did not have a case for loading raw data (which has only
    been tested for PV6.0.1) so I have added that in here. In addition, there
    were several things that were not working correctly for me or that I wanted
    options for. eg. read)bruker was returning 1980 points in my 2048 point file,
    which is because of Bruker having 68 points that you need to remove or leftshift
    but in the Matlab fidA code for Bruker, the leftshift is followed by a zero
    fill to bring back to the original 2048 size. So I've set that up as the 
    default here, although it can be changed.

    Parameters
    ----------
    fname : TYPE
        DESCRIPTION.
    fill_truncated_data : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    out1 : TYPE
        DESCRIPTION.

    """
    from brukerapi.dataset import Dataset
    import spec2nii.bruker as brkr
    from nifti_mrs.create_nmrs import gen_nifti_mrs_hdr_ext, gen_nifti_mrs
    # I'm not sure if this is the right thing to check. Some previous ParaVision
    # versions seem to use the naming convention fid.raw. However, brukerapi.dataset
    # seems to just use the file stem so I think it is better to stick with that convention
    # This version of the rawdata file is only tested on PV6.0.1. My understanding
    # is that PV5.1 does not save receive coil data separately, so this case is
    # set to 1 receiver in the override_path
    module_path=os.path.dirname(sys.modules[__name__].__file__)
    if 'rawdata' in fname:
        bruker_properties_path2=os.path.join(module_path,'..','bruker_properties','bruker_properties_rawdata_CB.json')
        bruker_override_path2 = os.path.join(module_path,'..','bruker_properties','bruker_rawdata_override_CB.json')
    else:
        bruker_properties_path2=os.path.join(module_path,'..','pyFidA','bruker_properties','bruker_properties_CB.json')
        bruker_override_path2 = os.path.join(module_path,'..','pyFidA','bruker_properties','bruker_fid_override_CB.json')
    bruker_properties_path = pkg_resources.resource_filename('spec2nii', 'bruker_properties.json')
    bruker_fid_override_path = pkg_resources.resource_filename('spec2nii', 'bruker_fid_override.json')
    #d2=Dataset(fname,property_files=[bruker_properties_path2])
    d2=Dataset(fname,property_files=[bruker_override_path2,bruker_properties_path2],parameter_files=['method'])
    # For whatever reason, I can't get d2 into the correct shape for the rawdata case using the parameters
    # in the json. Seems easier to just do it afterward. It loads with shape_storage but this just seems
    # based on block_size and block_count.
    # It appears that the rawdata type in schemas.py does not utilize the permute
    # variable/operation, so I could permute manually to put the channels last,
    # which seems to be what Jamie suggests for ordering in Matlab. However,
    # the GE code already puts coils before averages so maybe it doesn't matter.
    # I've commented out for now so that it remains [t,coils,averages] but, if
    # I change, I'll have to remember to change dim_type in bruker_rawdata_override_CB.json
    #if d2.type=='rawdata':
        #d2.data=np.transpose(np.reshape(d2.data,[d2.channels,-1,d2.block_count]),[1,2,0])
        #d2.data=np.transpose(d2.data,[0,2,1])
        # Potentially I should remove singleton dimensions here and adjust dim_type
        # to match, but I'm also rethinking that whole thing right now.
    # Then this is basically copied from spec2nii.bruker._proc_dataset, but that requires args to set
    # the output filename. Also, this gives me flexibility to set some parameters for rawdata
    # since spec2nii is only set up to read 2dseq and fid, really.
    # merge 2dseq complex frame group if present
    # I've added in the part for when the 2dseq data is not complex (which seems
    # to be the case for svs in PV6.0.1. Not sure if 2dseq is really only 
    # intended for mrsi maybe?)
    if d2.is_complex and d2.type == '2dseq':
        d2 = brkr.FrameGroupMerger().merge(d2, 'FG_COMPLEX')
    # actually, if it's a 2dseq for spectroscopic data, you are loading the spectral
    # data, not the fid. so you need to fft. It will then be complex
    elif not d2.is_complex and d2.type=='2dseq':
        d2.data=fid_from_specs(np.flipud(d2.data))
        d2.is_complex=True
    # Bruker raw and fid data have "junk" at the start that needs to be cut out
    # In Matlab's io_loadspec_bruk for fidA, the data are zero-filled back to 
    # the original data size. That is the default behaviour here, but both the
    # leftshift and zero fill can be left out. Note that we do not set the flag
    # for leftshift to True because that is intended for processing operations
    # for first order phase correction, not the standard data correction.
    if (d2.type=='rawdata' or d2.type=='fid') and do_leftshift:
        # Remove points acquired before echo
        d2.data = d2.data[d2.points_prior_to_echo:, ...]
        # fid data appears to need to be conjugated for NIFTI-MRS convention
        d2.data= d2.data.conj()
        if fill_truncated_data:
            pad_vals=[[0,d2.points_prior_to_echo] if dct==0 else [0,0] for dct in range(len(d2.data.shape))]
            d2.data=np.pad(d2.data, pad_width=pad_vals)
    if d2.is_svs:
        data=d2.data
        # This can probably done more neatly with np.newaxis
        data = np.expand_dims(np.expand_dims(np.expand_dims(data, axis=0), axis=0), axis=0)
    elif d2.is_mrsi:
        data=d2.data
        # push the spectral dimension to position 2
        data = np.moveaxis(data, 0, 2)
        # add empty dimensions to push the spectral dimension to the 3rd index
        data = np.expand_dims(data, axis=2)
    else:
        data = d2.data
    
    # get properties
    properties = d2.to_dict()
    # Orientation information
    if d2.type == 'fid':
        orientation = brkr.NIFTIOrient(brkr._fid_affine_from_params(d2))
    # added this in to match fid type, but not sure if that is correct. Should only matter for mrsi?
    elif d2.type == 'rawdata':
        orientation = brkr.NIFTIOrient(brkr._fid_affine_from_params(d2))
    else:
        orientation = brkr.NIFTIOrient(np.reshape(np.array(properties['affine']), (4, 4)))
    # Meta data. Setting dump=True, whereas it was a command line argument in the original
    if d2.type == 'fid':
        meta = brkr._fid_meta(d2, dump=True)
    if d2.type == 'rawdata':
        meta = brkr._fid_meta(d2, dump=True)
    else:
        meta = brkr._2dseq_meta(d2, dump=True)
    # Dwelltime - original code in brukerapi's _proc_dataset call had a factor of 2
    # to resolve because, for some reason, dwell_s was being calculated as 1/sw_h/2
    # But I changed so that dwell_s is 1/sw_h, so you don't need the factor 2 that
    # was inexplicably in there (maybe it applies for non-spectroscopic data or
    # other versions of ParaVision???)
    dwelltime = d2.dwell_s
    # And then we do the stuff to make nii
    im1=gen_nifti_mrs_hdr_ext(data, dwelltime, meta, orientation.Q44, no_conj=True)
    # Then we convert nifti_mrs to fidA
    out1=nii_to_fidA(im1)
    #if dtype is fid or 2dseq then you may need to set some flags, eg. averaged=True
    if (d2.type=='fid' or d2.type=='2dseq'):
        try:
            if out1.fids.shape[out1.dims['averages']]==1:
                out1.flags['averaged']=True
        except IndexError:
            out1.flags['averaged']=True
    # This covers the case where you load the rawdata but there is only 1 receive channel
    if d2.type=='rawdata' and d2.channels==1:
        out1.dims['coils']=-1
        out1.dims['averages']=1
    # Then I need to do this for the refscan file. OR could just leave to the user
    # since your input argument is a file, not a directory (as it is in Matlab)
    return out1

if __name__ == '__main__':
    pass
