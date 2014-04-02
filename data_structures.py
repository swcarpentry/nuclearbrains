# Functions
from .. import signal_processing as sigp
import numpy as np
import scipy.io as si
import tables as tbls
import datetime as dt
import scipy as sp
from sklearn import preprocessing as skp
import pickle as pickle
import copy as copy
import nitime as nt
from .. import utils

# Classes
import datetime as dt
import pandas as pd
import pandas.tseries as ts
from pandas import date_range, DateOffset
from pandas.tseries.offsets import DateOffset
from pandas.tseries.index import DatetimeIndex

'''Defines the data structures that we can use
with ECoG.  Generally, this is just a nice way
to keep metadata along with the signal data itself.
'''

__all__ = ['convert_sec_to_ix',
           'find_locs_2d',
           'slice_seconds',
           'index_from_array_shape',
           'set_vals_to',
           'create_df_from_trial_secs',
           'Ecog_df',
           'Audio_df',
           'ModelResults',
           'StimInfo',
           'Params'
           ]

def convert_sec_to_ix(seconds, s_rate):
    '''Takes a list of seconds, converts each of them to an index based on s_rate'''
    return np.array([int(second*s_rate) for second in seconds])

def find_locs_2d(locfile):
    '''Locfile is a dictionary of 'electrode: (x, y)' location pairs.  It will turn this into a dataframe so that
    we can add it to an ecog class.'''
    xycoords = pickle.load(open(locfile))
    return pd.DataFrame(np.array(xycoords.values()), columns=['x_2d', 'y_2d'], index=np.array(xycoords.keys())+1)


#### GENERAL CONTAINER CLASSES ####
class Params():
    '''This class will hold the parameters of our ecog array'''
    def __init__(self):
        pass

class StimInfo():
    '''This class will hold stimulus information of the ecog array'''
    def __init__(self):
        pass


######## CLASS USING DATAFRAMES ########

def slice_seconds(s=0, f=2, period=1000):
    '''This returns a datetime slice in seconds'''
    start = dt.datetime(2000, 1, 1, microsecond=0)
    time1 = start + pd.DateOffset(seconds=s)
    time2 = time1 + pd.DateOffset(seconds=f, period=period)
    return [time1, time2]

def index_from_array_shape(array, start=dt.datetime(2000, 1, 1, microsecond=0), period=1000):
    '''Takes an array as input, creates a daterange of length array.shape[0] with sampling frequency of freq
    freq is in microseconds'''
    length = array.shape[0]
    ix_period =np.arange(length) * period
    return pd.to_datetime(ix_period, unit='us')  

def sampling_rate_to_microseconds(s_rate):
    '''Converts a sampling rate to a period in microseconds'''
    p_sec = 1./s_rate * 1e6
    return int(np.ceil(p_sec))

def set_vals_to(signal, vals, kind='median'):
    '''
    Sets values of a certain type to a standard number.  This is useful if you want to minimize
    the impact of extreme values, but don't want to remove the datapoint itself
    
    INPUTS:
        signal          :   (dataframe) the signal that has values you wish to replace
        vals            :   (index or mask) the value of the index that you wish to replace,
                            or a mask that identifies indices to replace
        kind            :   The new value to use in repacement. Must be "median" or an int
    '''
    if kind=='median':
        nval = np.median(signal, 0)
    else:
        try:
            nval = [kind]*len(signal.columns)
        except NameError: 
            print('You need to specify "median", or choose an integer')
    for i, col in enumerate(signal[vals].columns):
        signal.loc[vals, col] = nval[i]

def create_df_from_trial_secs(trials, signal_ref='None', s_rate=1000):
    '''This takes an input of an array with 3 columns: trial begin (s), trial end (s), and trial type.
    It returns a dataframe corresponding to the length of the session given some sampling rate, 
    where each point is labeled by a trial type.

    Note that it expects trials to be in triplets.

    INPUTS:
        trials          :   array with 3 columns: trial begin (s), trial end (s), and trial type (an integer)
        signal_ref      :   output column will be the same length as this signal
        s_rate          :   the sampling rate of the signal - this affects how time is handled

    OUTPUTS:
        new_df          :   A pandas dataframe with three columns:
                                1. Trial type, an integer representation of trial types
                                2. Trial number, the trial number, aka number of times that trial type has been found
                                3. Timebase, an arbitrary representation of time that begins at 0
        '''
    # Define the ixs we want to use and set up dummy structures
    df = pd.DataFrame(trials, copy=True)
    if signal_ref != 'None':
        # In case the reference signal has extra timepoints
        print('Matching to signal')
        assert signal_ref.shape[0] > int(s_rate*df.iloc[-1, 1]), 'Reference signal is too short'
        ix_end = signal_ref.shape[0]
    else:
        # No reference signal, so we decide length based on s_rate
        ix_end = int(s_rate*df.iloc[-1, 1])
        

    new_df = np.ones(ix_end*3).reshape(ix_end, 3) * -1
    #Convert seconds to indices according to sampling rate
    to_ix = lambda second: int(second*s_rate)
    df.iloc[:, :2] = df.iloc[:, :2].applymap(to_ix)

    # Now convert parts of the dummy structure to numbers based on classes in "trials"
    trialnum = -1
    for ix, (trial_s, trial_e, trial_t) in df.astype(int).iterrows():
        if trial_t==0: trialnum+=1
        trialslice = slice(trial_s, trial_e)

        # Insert trial type, trial number, and timebase within a trial
        new_df[trialslice, 0] = trial_t
        new_df[trialslice, 1] = trialnum
        new_df[trialslice, 2] = np.arange(trial_e-trial_s)

    new_df = pd.DataFrame(new_df,
                            columns=['t_type', 't_num', 't_base'],
                            index=nt.UniformTime(length=new_df.shape[0], sampling_rate=s_rate))
                        
    return new_df

#### General DS Container
class DataStructure:
    def __init__(self, signal, srate, filename, savepath, time_index=True):
        # Initialize containers for data
        # Contains information about stimuli such as timings
        self.stim_info = StimInfo()

        # Parameters for the signal such as sampling rate
        self.params = Params()
        self.params_db = {}     # Keeps track of what parameter pickle files exist

        # Keeps track of whatever processing steps we've used on this dataset
        self.params.proc_hist = []

        # Tells us the location and type of the signal
        self.params.signal_type = 'raw'

        # Where will we save this dataset, and what will we call it?
        self.params.savepath = savepath
        self.params.fname = filename
        self.params.hdfloc = savepath + filename + '_data.h5'
        self.params.paramloc = savepath + filename + '_params.p'

        # Signal-specific
        self.params.s_rate = srate

        # Initialize signal, metadata, and timebase
        self.time = nt.UniformTime(length=signal.shape[0], sampling_rate=srate)
        self.signal = pd.DataFrame(signal, columns=np.arange(signal.shape[1])+1)
        if time_index==True: 
            self.signal.set_index(self.time, inplace=True)
        self.signalmetadata = pd.DataFrame(index=self.signal.index)
        self.colmetadata = pd.DataFrame(index=self.signal.columns)

    def downsample(self, down_fac):
        self.signal = sigp.downsample_df(self.signal, down_fac)
        self.signalmetadata = self.signalmetadata.loc[::down_fac, :].reset_index(drop=True)

    def filter_by_condition(self, cond_list):
        return  [self.signal.loc[(self.signalmetadata.t_type==cond).values] for cond in cond_list]
        
    def load_hdf(self, d_type):
        '''Loads data contained in the HDF corresponding to an instance of this object.
        Data type is specified in d_type, and must be a field that exists in params_db
        '''
        print('Loading data type: {0}'.format(d_type))
        try:
            loc = self.params_db[d_type].hdfloc
        except KeyError:
            print("That data doesn't exist, create it first or choose another data type")
        
        self.signal = pd.read_hdf(loc, d_type)

    def save_ds(self):
        '''Saves an ECoG datastructure to an HDF5 file (for the data) as well as a pickle file (for subject information)
        '''
        ds = copy.deepcopy(self)
        print('Saving data type: {0}\ntolocation: {1}'.format(ds.params.signal_type, ds.params.hdfloc))
        # Now save the HDFs to the correct name
        ds.signal.to_hdf(ds.params.hdfloc, '/{0}'.format(ds.params.signal_type))
        del ds.signal
        
        # Save parameters to a dictionary that keeps parameters of interest
        ds.params_db[ds.params.signal_type] = ds.params

        with open(ds.params.paramloc, 'w') as f:
            pickle.dump(ds, f)
            f.close() 



### Classes we'll use for the dataframes

class Ecog_df(DataStructure):
    '''Main class for representing ecog data using pandas dataframes.  
        The most important non-obvious pieces of this are:
            data            :   The tables object representing the HDF5 data
            signal          :   A dataframe with the signal 
            params          :   Important parameters for this ecog dataset
                proc_hist   :   A list of all processing that has been done on the signal (e.g., filters)
            stim_info       :   Holds information about the stimuli that were presented
    '''
    def __init__(self, *args, **kwargs):
        DataStructure.__init__(self, *args, **kwargs)
        self.colmetadata['grpng'] = np.ones(self.signal.columns[-1])
        self.colmetadata['elec_good'] = np.ones(self.signal.columns[-1]).astype(bool)

    def set_grpng(self, grpdict):
        for grp, ixs in grpdict.iteritems():
            self.colmetadata.grpng.iloc[ixs] = grp

    def set_bad(self, bdelecs):
        self.colmetadata.elec_good[bdelecs] = 0  

    def find_extreme_timepoints(self, cutoff=5):
        '''
        Takes the z-score of each electrode and returns a binary vector that indexes which
        timepoints are OK for each electrode.

        INPUTS:
            cutoff          :   How many standard deviations must a point be to be 'bad'  
        '''
        # Find timepoints that lie outside of 'cutoff' scores away
        zsig = sp.stats.zscore(self.signal)
        badixs = np.any(abs(zsig) > cutoff, axis=1)
        percbad = (badixs.sum().astype(float) / self.signal.shape[0]) * 100
        print('Signal falling outside {0} STDs: {1}%'.format(cutoff, percbad))

        # Now take the inverse of the indices of bad timepoints. A bit convoluted, but
        # This is a bit more intuitive considering the
        self.signalmetadata['good_time_ixs'] = (badixs == False)
    
    def load_results(self, fname='model_results.p'):
        results = utils.quickload(self.params.savepath + fname)
        results.show_modelresult_models()
        return results

    def _extract_good_elecs(self):
        self.signal =  self.signal.loc[:, self.colmetadata.elec_good].squeeze()
        self.colmetadata.grpng = self.colmetadata.grpng * self.colmetadata.elec_good
            
    ## Batch functions
    def preprocess(self):
        '''A general method to run a bunch of preprocessing methods in one swoop'''
        print('Running simple preprocessing\n----------------------------')

        print('Extracting good electrodes')
        self._extract_good_elecs()

        print('Highpass filter...')
        if not hasattr(self.params, 'elec_grp'): self.params.elec_grp = np.ones(self.signal.shape[1])


        self.signal = self.signal.apply(sigp.filtersignal, axis=0, 
                          s_rate=self.params.s_rate, f_type='highpass', lims=[.5])
        self.params.proc_hist.append('highpass')

        print('Applying 3 notch filters...')
        self.signal = self.signal.apply(sigp.notch_all, s_rate=self.params.s_rate)
        self.params.proc_hist.append('notch_all')

        print('Applying CAR...')
        self.signal = sigp.car(self.signal, self.colmetadata.grpng.values)
        self.params.proc_hist.append('CAR')

        print('Finding extreme timepoints')
        self.find_extreme_timepoints()



class Audio_df(DataStructure):
    '''Main class for representing audio data using pandas dataframes.  
        The most important non-obvious pieces of this are:
            data            :   The tables object representing the HDF5 data
            signal_good     :   The signal with good electrodes selected out
            signal_proc     :   A dataframe with the signal that has been processed in any way 
            params          :   Important parameters for this ecog dataset
                proc_hist   :   A list of all processing that has been done on the signal (e.g., filters)
            stim_info       :   Holds information about the stimuli that were presented
    '''
    def __init__(self, *args, **kwargs):
        DataStructure.__init__(self, *args, **kwargs)
        
    ## Batch functions
    def preprocess(self, scale=True):
        '''A general method to run a bunch of preprocessing methods in one swoop'''
        pass

class ModelResults():
    '''
    A container to store model results and parameters.  Currently, it has a single attribute
    which is meant to be a dictionary with lists of three fields: model type, model parameters, and
    model score
    '''
    def __init__(self, savepath):
        self.modeldict = {}
        self.savepath = savepath
        self.fname = 'model_results'

    def add_model(self, mname, mparams, mscores, modelnotes=None):
        '''
        Adds a new model to the modeldict dictionary.

        INPUTS:
            mname           :   The name of the model.  This becomes the key in modeldict
            mparams         :   a pandas dataframe of model parameters
            mscores         :   a pandas dataframe of model scores
            modelnotes      :   any extra notes, defaults to 'None'
        '''
        if mname in self.modeldict.keys():
            cont = raw_input('Model already contains this item, overwrite? (y/n): ')
            if cont == 'y':
                print('Overwriting...')
            else:
                raise NameError('Please rename your model')

        self.modeldict[mname] = {'params':mparams, 'scores':mscores, 'notes':modelnotes}
        print('Added model item: {0}'.format(mname))

    def remove_model(self, mname):
        '''
        Remove a model from modeldict
        '''
        try:
            self.modeldict.pop(mname)
            print("Removed model info: {0}".format(mname))
        except KeyError as myerror:
            print("This model name doesn't exist: {0}".format(myerror))

    def show_modelresult_models(self, mname='None'):
        '''
        Shows the results in a model object
        '''
        for key, value in self.modeldict.iteritems():
            if (mname == 'None') | (mname in key):
                print key
                
                for key, value in value.iteritems():
                    print '----|', key
                    
                    if isinstance(value, dict) == True:
                        for key, value in value.iteritems():
                            print '----|----|', key
                print '\n'

    def save_results(self):
        '''
        Saves the object to its save location
        '''
        utils.quickdump(self, self.savepath + self.fname + '.p')