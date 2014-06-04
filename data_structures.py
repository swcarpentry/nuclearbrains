# Functions
from .. import signal_processing as sigp
from .. import io
import numpy as np
import scipy.io as si
import tables as tbls
import datetime as dt
import scipy as sp
from sklearn import preprocessing as skp
import cPickle as pickle
import copy as copy
import nitime as nt
import mne
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
           'index_from_array_shape',
           'set_vals_to',
           'create_df_from_trial_secs',
           'Ecog_df',
           'Audio_df',
           'ModelResults',
           'TimeInfo',
           'Params'
           ]


def convert_sec_to_ix(seconds, s_rate):
    '''Takes a list of seconds, converts each of them to an index
    based on s_rate'''
    return np.array([int(second*s_rate) for second in seconds])


def find_locs_2d(locfile):
    '''Locfile is a dictionary of 'electrode: (x, y)' location pairs.  It will
    turn this into a dataframe so that
    we can add it to an ecog class.'''
    xycoords = pickle.load(open(locfile))
    return pd.DataFrame(np.array(xycoords.values()), columns=['x_2d', 'y_2d'],
                        index=np.array(xycoords.keys())+1)


# GENERAL CONTAINER CLASSES ####
class Params():
    '''This class will hold the parameters of our ecog array'''
    def __init__(self):
        pass


class TimeInfo():
    '''This class will hold information about timepoints in a signal'''
    def __init__(self):
        pass


# ----- CLASS USING DATAFRAMES ----
def index_from_array_shape(array, start=dt.datetime(2000, 1, 1, microsecond=0),
                           period=1000):
    '''Takes an array as input, creates a daterange of length array.shape[0]
    with sampling frequency of freq

    freq is in microseconds'''
    length = array.shape[0]
    ix_period = np.arange(length) * period
    return pd.to_datetime(ix_period, unit='us')


def sampling_rate_to_microseconds(s_rate):
    '''Converts a sampling rate to a period in microseconds'''
    p_sec = 1./s_rate * 1e6
    return int(np.ceil(p_sec))


def set_vals_to(signal, vals, kind='median'):
    '''
    Sets values of a certain type to a standard number.  This is useful
    if you want to minimize the impact of extreme values, but don't want
    to remove the datapoint itself

    INPUTS:
        signal          :   (dataframe) the signal that has values you wish
                                to replace
        vals            :   (index or mask) the value of the index that you
                                wish to replace,
                            or a mask that identifies indices to replace
        kind            :   The new value to use in repacement. Must be
                                "median" or an int
    '''
    if kind is 'median':
        nval = np.median(signal, 0)
    else:
        try:
            nval = [kind]*len(signal.columns)
        except NameError:
            print('You need to specify "median", or choose an integer')
    for i, col in enumerate(signal[vals].columns):
        signal.loc[vals, col] = nval[i]


def create_df_from_trial_secs(df, signal_ref=None, s_rate=1000,
                              time_index=False, timebase_col=None,
                              docut=False):
    '''This takes an input of an array with 3 columns: trial begin (s),
    trial end (s), and trial type. It returns a dataframe corresponding
    to the length of the session given some sampling rate, where each
    point is labeled by a trial type.

    Note that it expects trials to be in triplets.

    INPUTS:
        df              :   a pandas dataframe with at least three columns:
                            'start' and 'stop', denoting the beginning/end of
                            a trial (in seconds), and any number of numeric
                            columns that represent different features. Each
                            row corresponds to a trial
        signal_ref      :   output column will be the same length
                            as this signal
        s_rate          :   the sampling rate of the signal - this affects
                            how time is handled

    OUTPUTS:
        new_df          :   A pandas dataframe with columns equal to the
                            number of features (other than start/stop) that
                            you put in. The rows will now be expanded according
                            to the times provided, as well as the sampling rate
                            of the signal
    '''
    assert isinstance(df, pd.DataFrame), 'input needs to be a dataframe'
    df = df.copy()
    timebase_col = ["None"] if timebase_col is None else list(timebase_col)

    # Replace "end" with the maximum value in the stop row
    # to avoid it being an "object" row
    if 'end' in df['stop'].values:
        maxend = df['stop'].replace('end', 0).max()
        df['stop'] = df['stop'].replace('end', maxend).astype(float)

    # Remove non-numeric columns from the df (because it really slows things
    # down to copy objects)
    for nm, vals in df.iteritems():
        try:
            vals.dropna().astype(int)
        except ValueError:
            print('Dropping non-numeric column: {0}'.format(nm))
            df.drop(nm, axis=1, inplace=True)

    # Define the times / features separately
    times = df[['start', 'stop']]
    feats = df.drop(['start', 'stop'], 1)
    nitems, nfeats = feats.shape
    nfeats += 1  # For the timebase
    fnames = list(feats.columns)
    fnames.append('t_base')

    # Define the ixs we want to use and set up dummy structures
    newlen = int(s_rate*df['stop'].max())
    if signal_ref is not None:
        # In case the reference signal has extra timepoints
        reflen = signal_ref.shape[0]
        if docut is False:
            assert reflen > newlen, 'Reference signal is too short. Use "docut" \
                                    to override this. Ref is {0}, \
                                    new is {1}'.format(reflen, newlen)
        ix_end = reflen
    else:
        # No reference signal, so we decide length based on s_rate
        ix_end = newlen

    # Convert seconds to indices according to sampling rate
    to_ix = lambda second: int(second*s_rate)
    times = times.applymap(to_ix)

    # Convert parts of the dummy structure to numbers
    # based on classes in "trials"
    feat_dict = {name: np.ones([ix_end])*-1 for name in fnames}
    for i in range(nitems):
        # Pull times
        trial_s, trial_e = times.values[i, :]
        trialslice = slice(trial_s, trial_e)

        # Now insert features
        # Insert trial type, trial number, and timebase within a trial.
        # -1 for the timebase
        floopnames = (feat for feat in fnames if feat is not 't_base')
        for j, name in enumerate(floopnames):
            val = feats.iloc[i, j]
            if bool(np.isnan(val)) is not True:
                feat_dict[name][trialslice] = feats.iloc[i, j]
                if name in timebase_col:
                    feat_dict['t_base'][trialslice] = np.arange(trial_e-trial_s)

    # Now add the timebase
    myix = nt.UniformTime(length=ix_end,
                          sampling_rate=s_rate) if time_index is True else None
    new_df = pd.DataFrame(feat_dict, index=myix)

    return new_df


# ---- General DS Container ----
class DataStructure(object):
    def __init__(self, signal, s_rate,
                 time_info=None, colmetadata=None, params=None,
                 params_uniq=None, incols=None, signal_type=None):

        # Initialize dataframe containers for data
        # Contains information about stimuli such as timings
        incols = np.arange(signal.shape[1])+1 if incols is None else incols
        self.params = pd.DataFrame() if params is None else params
        self.time_info = pd.DataFrame() if time_info is None else time_info
        self.colmetadata = pd.DataFrame(index=incols) if colmetadata is None else colmetadata

        if params_uniq is None:
            self.params_uniq = {'proc_hist': [], 's_rate': int(s_rate)}
            self.params_uniq['signal_type'] = 'raw' if signal_type is None else signal_type
        self.s_rate = s_rate

        # Initialize signal, metadata, and timebase
        self.signal = pd.DataFrame(signal, columns=incols)

    def downsample(self, down_fac):
        self.signal = sigp.downsample_df(self.signal, down_fac)
        self.s_rate /= down_fac
        self.params_uniq['s_rate'] = self.s_rate

    def filter_by_condition(self, cond_dict):
        mask = utils.mask_df(self.signalmetadata, cond_dict)
        return self.signal[mask]

    @staticmethod
    def _load_hdf_params(hdfloc, d_type):
        '''Loads data contained in the HDF corresponding to an instance
        of this object. Data type is specified in d_type, and must be a
        field that exists in params_db.
        '''

        # First load the signal dataframe
        try:
            signal = pd.read_hdf(hdfloc, d_type+'/signal')
        except TypeError:
            print('Trying to load array directly')
            with tbls.open_file(hdfloc, 'r') as h:
                signal = h.get_node('/{0}/signal'.format(d_type)).read()
        except KeyError:
            print("That data doesn't exist, create it first or choose another\
                  data type")

        # Now load metadata
        with tbls.open_file(hdfloc, 'r') as h:
            attrs = h.get_node('/{0}/signal/'.format(d_type))._v_attrs

        params_uniq = {'param_loc': attrs.param_loc,
                       's_rate': attrs.s_rate,
                       'signal_type': d_type}

        # Finally, read in the metadata structures
        save_name = attrs.save_name
        save_path = attrs.save_path
        colmetadata = pd.read_csv(save_path+'{0}_colmetadata.csv'.format(save_name), index_col=0)
        time_info = pd.read_csv(save_path+'{0}_time_info.csv'.format(save_name), index_col=0)
        params = pd.read_csv(save_path+'{0}_params.csv'.format(save_name), index_col=0)

        return {'signal': signal,
                'colmetadata': colmetadata,
                'time_info': time_info,
                'params': params,
                'params_uniq': params_uniq}

    def gen_metadata_from_time_info(self, time_info=None, signal_ref=None,
                                    timebase_col='t_type', **kwargs):
        '''
        Generates a signalmetadata field from trialtrialinfo. This must be
        located in time_info.stims, and is a dataframe that conforms to the
        specifications given in create_df_from_trial_secs
        '''
        stims = self.time_info if time_info is None else time_info
        signal_ref = self.signal if signal_ref is None else signal_ref
        self.signalmetadata = create_df_from_trial_secs(
            stims, signal_ref, s_rate=self.s_rate,
            timebase_col=timebase_col, **kwargs
            )

    def save_ds(self, save_path, save_name):
        '''Saves an ECoG datastructure to an HDF5 file (for the data) as well
        as a pickle file (for subject information)

        PARAMETERS:
        save_path            :   The path we'll use when saving file. type str
        save_name            :   The name we'll use when saving files. type str
        '''

        stype = self.params_uniq['signal_type']
        hpath = save_path + save_name + '_data.h5'
        hdfkey = '/{0}/'.format(stype)
        print('Saving data type: {0}\nto location: {1}'.format(stype, hpath))

        # Save all metadata to csv files and signal to hdf
        self.colmetadata.to_csv(save_path+'{0}_colmetadata.csv'.format(save_name))
        self.time_info.to_csv(save_path+'{0}_time_info.csv'.format(save_name))
        self.params.to_csv(save_path+'{0}_params.csv'.format(save_name))
        self.signal.to_hdf(hpath, hdfkey+'signal', mode='a')

        # Save metadata/parameters to the hdf file.
        f = tbls.open_file(hpath, 'a')
        save_dict = {'s_rate': self.s_rate,
                     'proc_hist': self.params_uniq['proc_hist'],
                     'param_loc': save_path,
                     'save_name': save_name,
                     'save_path': save_path}
        for key, val in save_dict.iteritems():
            f.set_node_attr(hdfkey+'signal', key, val)
        f.close()

    def to_mne(self, ch_types='eeg'):
        '''
        Returns the MNE.raw object of our data
        '''
        nchans = self.signal.shape[1]
        ch_types = [ch_types] * nchans
        nms = [str(i) for i in range(nchans)]
        info = mne.io.array.create_info(nms, self.s_rate, ch_types)
        raw = mne.io.array.RawArray(self.signal.T.values, info)
        return raw


# ----Classes we'll use for the dataframes----
class Ecog_df(DataStructure):
    '''Main class for representing ecog data using pandas dataframes.
        The most important non-obvious pieces of this are:

        PARAMETERS:
        signal              :   The ecog signal to be saved. Time x features
        srate               :   The sampling rate of the signal. type Int
        time_info=None      :   A structure for saving time information.
                                    type DataFrame
        colmetadata=None    :   A structure for storing column metadata.
                                    type DataFrame
        params=None         :   A structure for storing general ecog
                                    parameters. type DataFrame
        incols=None         :   A vector to specify naming for the columns
        signal_type=None    :   If we want to specify a custom signal type.
                                    type str
    '''
    def __init__(self, *args, **kwargs):
        DataStructure.__init__(self, *args, **kwargs)
        if 'grouping' not in self.colmetadata.columns:
            self.colmetadata['grouping'] = -1
        if 'good_elecs' not in self.colmetadata.columns:
            self.colmetadata['good_elecs'] = True

    def set_grouping(self, grpdict):
        for grp, ixs in grpdict.iteritems():
            # Make indices 0-indexed
            ixs = [i-1 for i in ixs]
            self.colmetadata.grouping.iloc[ixs] = grp

    def set_bad(self, bdelecs):
        self.colmetadata.elec_good[bdelecs] = 0

    def find_extreme_timepoints(self, cutoff=5):
        '''
        Takes the z-score of each electrode and returns a binary vector
        that indexes which timepoints are OK for each electrode.

        INPUTS:
            cutoff          :   How many standard deviations must a point
            be to be 'bad'
        '''
        # Find timepoints that lie outside of 'cutoff' scores away
        zsig = sp.stats.zscore(self.signal)
        badixs = np.any(abs(zsig) > cutoff, axis=1)
        percbad = (badixs.sum().astype(float) / self.signal.shape[0]) * 100
        print('Signal falling outside {0} STDs: {1}%'.format(cutoff, percbad))

        # Now take the inverse of the indices of bad timepoints.
        self.signalmetadata['extreme_points'] = badixs

    @classmethod
    def from_hdf(cls, hdfloc, d_type):
        '''Loads data contained in the HDF corresponding to an
        instance of this object. Data type is specified in d_type,
        and must be a field that exists in params_db

        These are the parameters that _load_hdf_params gives you:

        Hdf params dict:
        {'signal',
        'colmetadata':colmetadata,
        'time_info':time_info,
        'params':params,
        'params_uniq':params_uniq}

        Params uniq:
        {'param_loc': attrs.param_loc,
        's_rate':attrs.s_rate,
        'signal_type':d_type}
        '''
        print('Loading data type: {0}'.format(d_type))
        dd = DataStructure._load_hdf_params(hdfloc, d_type)
        puniq = dd['params_uniq']

        # Construct the data structure
        ld = {nm: dd[nm] for nm in ['time_info', 'colmetadata', 'params']}
        ld['signal_type'] = puniq['signal_type']

        return cls(dd['signal'], puniq['s_rate'], **ld)

    def load_results(self, dict_path=None, dict_name='model_results.p'):
        if dict_path is None:
            dict_path = self.params.savepath
        return ModelResults.from_dict(dict_path, dict_name)

    def _extract_good_elecs(self):
        self.signal = self.signal.loc[:, self.colmetadata.good_elecs.astype(bool)].squeeze()
        self.colmetadata.grouping = self.colmetadata.grouping * self.colmetadata.good_elecs

    # Batch functions
    def preprocess(self):
        '''A general method to run a bunch of preprocessing
        methods in one swoop'''
        print('Running simple preprocessing\n----------------------------')

        print('Extracting good electrodes')
        self._extract_good_elecs()

        print('Highpass filter...')
        if not hasattr(self.params, 'grouping'):
            self.params.grouping = np.ones(self.signal.shape[1])

        self.signal = self.signal.apply(sigp.filtersignal, axis=0,
                                        s_rate=self.s_rate, f_type='highpass',
                                        lims=[.5])
        self.params_uniq['proc_hist'].append('highpass')

        print('Applying 3 notch filters...')
        self.signal = self.signal.apply(sigp.notch_all, s_rate=self.s_rate)
        self.params_uniq['proc_hist'].append('notch_all')

        print('Applying CAR...')
        self.signal = sigp.car(self.signal, self.colmetadata.grouping.values)
        self.params_uniq['proc_hist'].append('CAR')

        print('Finding extreme timepoints')
        self.find_extreme_timepoints()

        self.params_uniq['signal_type'] = 'preproc'


class Audio_df(DataStructure):
    '''Main class for representing audio data using pandas dataframes.
            The most important non-obvious pieces of this are:
            data            :   The tables object representing the HDF5 data
            signal_good     :   The signal with good electrodes selected out
            signal_proc     :   A dataframe with the signal that has been
                                    processed in any way
            params          :   Important parameters for this ecog dataset
                proc_hist   :   A list of all processing that has been done on
                                    the signal (e.g., filters)
            time_info       :   Holds information about the stimuli that were
                                    presented
    '''
    def __init__(self, *args, **kwargs):
        DataStructure.__init__(self, *args, **kwargs)

    @classmethod
    def from_hdf(cls, hdfloc, d_type):
        '''Loads data contained in the HDF corresponding to an instance
        of this object. Data type is specified in d_type, and must be a
        field that exists in params_db

        These are the parameters that _load_hdf_params gives you:

        Hdf params dict:
        {'signal',
        'colmetadata':colmetadata,
        'time_info':time_info,
        'params':params,
        'params_uniq':params_uniq}

        Params uniq:
        {'param_loc': attrs.param_loc,
        's_rate':attrs.s_rate,
        'signal_type':d_type,
        'proc_hist':attrs.proc_hist}
        '''
        print('Loading data type: {0}'.format(d_type))
        dd = DataStructure._load_hdf_params(hdfloc, d_type)
        puniq = dd['params_uniq']

        # Construct the data structure
        ld = {nm: dd[nm] for nm in ['time_info', 'colmetadata', 'params']}
        ld['signal_type'] = puniq['signal_type']

        return cls(dd['signal'], puniq['s_rate'], **ld)


class ModelResults():
    '''
    A container to store model results and parameters.  Currently, it has
    a single attribute which is meant to be a dictionary with lists of
    three fields: model type, model parameters, and model score
    '''
    def __init__(self, file_name='model_results', modeldict=None):
        self.modeldict = {} if modeldict is None else modeldict
        self.file_name = file_name

    def __repr__(self):
        return self.show_modelresult_models(doprint=False)

    @classmethod
    def from_dict(cls, dict_path, dict_name='model_results.p'):
        modeldict = io.quickload(dict_path + dict_name)
        return cls(file_name=dict_name, modeldict=modeldict)

    def add_model(self, mname, mparams, mscores, modelnotes=None):
        '''
        Adds a new model to the modeldict dictionary.

        INPUTS:
            mname           :   The name of the model.  This becomes the
                                    key in modeldict
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

        self.modeldict[mname] = {'params': mparams,
                                 'scores': mscores,
                                 'notes': modelnotes}
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

    def show_modelresult_models(self, mname=None, doprint=True):
        '''
        Shows the results in a model object
        '''
        sl = []
        for key, value in self.modeldict.iteritems():
            if (mname is None) or (mname in key):
                sl.append(key)

                for key, value in value.iteritems():
                    sl.append('\n')
                    sl.append('----|'+key)

                    if isinstance(value, dict) is True:
                        for key, value in value.iteritems():
                            sl.append('\n')
                            sl.append('----|'*2 + key)
                sl.append('\n')
        s = ''.join(sl)
        if doprint is True:
            print s
        else:
            return s

    def save_results(self, save_path, save_name=None):
        '''
        Saves the object to its save location
        '''
        if save_name is None:
            save_name = self.file_name
        io.quickdump(self.modeldict, save_path + save_name + '.p')