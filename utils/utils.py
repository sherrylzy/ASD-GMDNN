import os
import time
import warnings
import numpy as np

import h5py
import torch
import shutil
from collections import OrderedDict
import matplotlib.pyplot as plt
import collections
import tqdm
import librosa
import librosa.core
import librosa.feature
import yaml
import torchaudio
import torch
from tqdm import tqdm


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)


def save_checkpoint(state, is_best, task_id, model_file, filename='checkpoint.pth.tar'):
    torch.save(state, model_file + '/' + task_id + filename)
    if is_best:
        shutil.copyfile(model_file + '/' + task_id + filename, task_id + 'model_best.pth.tar')


def save_checkpoint_multigpu(state, is_best, task_id, model_file, filename='checkpoint.pth.tar'):
    torch.save(state, model_file + '/' + task_id + filename)
    if is_best:
        shutil.copyfile(model_file + '/' + task_id + filename, task_id + 'model_best.pth.tar')


"""
From http://wiki.scipy.org/Cookbook/SegmentAxis
"""


def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis into overlapping frames.

    example:
    >>> segment_axis(np.arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    arguments:
    a       The array to segment
    length  The length of each frame
    overlap The number of array elements by which the frames should overlap
    axis    The axis to operate on; if None, act on the flattened array
    end     What to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:

            'cut'   Simply discard the extra values
            'wrap'  Copy values from the beginning of the array
            'pad'   Pad with a constant value

    endvalue    The value to use for end='pad'

    The array is not copied unless necessary (either because it is
    unevenly strided and being flattened or because end is set to
    'pad' or 'wrap').
    """

    if axis is None:
        a = np.ravel(a)  # may copy
        axis = 0

    l = a.shape[axis]

    if overlap >= length: raise ValueError(
        "frames cannot overlap by more than 100%")
    if overlap < 0 or length <= 0: raise ValueError(
        "overlap must be nonnegative and length must be positive")

    if l < length or (l - length) % (length - overlap):
        if l > length:
            roundup = length + (1 + (l - length) // (length - overlap)) * (
                    length - overlap)
            rounddown = length + ((l - length) // (length - overlap)) * (
                    length - overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown < l < roundup
        assert roundup == rounddown + (length - overlap) or (
                roundup == length and rounddown == 0)
        a = a.swapaxes(-1, axis)

        if end == 'cut':
            a = a[..., :rounddown]
        elif end in ['pad', 'wrap']:  # copying will be necessary
            s = list(a.shape)
            s[-1] = roundup
            b = np.empty(s, dtype=a.dtype)
            b[..., :l] = a
            if end == 'pad':
                b[..., l:] = endvalue
            elif end == 'wrap':
                b[..., l:] = a[..., :roundup - l]
            a = b

        a = a.swapaxes(-1, axis)

    l = a.shape[axis]
    if l == 0: raise ValueError(
        "Not enough data points to segment array in 'cut' mode; "
        "try 'pad' or 'wrap'")
    assert l >= length
    assert (l - length) % (length - overlap) == 0
    n = 1 + (l - length) // (length - overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n, length) + a.shape[axis + 1:]
    newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[
                                                                  axis + 1:]

    if not a.flags.contiguous:
        a = a.copy()
        newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[
                                                                      axis + 1:]
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)

    try:
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)
    except TypeError or ValueError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[
                                                                      axis + 1:]
        return np.ndarray.__new__(np.ndarray, strides=newstrides,
                                  shape=newshape, buffer=a, dtype=a.dtype)


def mkdir_p(path):
    """ Creates a path recursively without throwing an error if it already exists

    :param path: path to create
    :return: None
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    except FileNotFoundError:
        if path == '':
            pass
class MyOrderedDict(OrderedDict):
    def prepend(self, key, value, dict_setitem=dict.__setitem__):

        root = self._OrderedDict__root
        first = root[1]

        if key in self:
            link = self._OrderedDict__map[key]
            link_prev, link_next, _ = link
            link_prev[1] = link_next
            link_next[0] = link_prev
            link[0] = root
            link[1] = first
            root[1] = first[0] = link
        else:
            root[1] = first[0] = self._OrderedDict__map[key] = [root, first, key]
            dict_setitem(self, key, value)


def plot_loss_moment(losses, hyp):
    _, ax = plt.subplots(figsize=(16, 9), dpi=80)
    ax.plot(losses, 'blue', label='train', linewidth=1)
    ax.set_title('Loss change in training')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    ax.legend(loc='upper right')
    plt.savefig(os.path.join(hyp))


def load_torchmodel(model, file_path):
        try:
            model.load_state_dict(torch.load(file_path))
        except Exception as e:
            print('Failed to load model: %s' % (e))
            exit(1)
        return model



def file_to_AST(file_name,
                n_mels=128,
                n_fft=1024):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """

    # generate LMFB from torchaudio. Here the y is tensor, but in baseline, the y is 'ndarry' type.
    y, sr = torchaudio.load(file_name)
    #y, sr = file_load(file_name, mono=True)
    fbank = torchaudio.compliance.kaldi.fbank(
        y, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=n_mels, dither=0.0,
        frame_shift=10)
    # window_type (str, optional): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman')
    n_frames = fbank.shape[0]

    p = n_fft - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:n_fft, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    fbank = fbank.numpy()

    '''
    # calculate total vector size
    n_vectors = len(fbank[0, :]) - n_frames + 1

    # skip too short clips
    if n_vectors < 1:
        return np.empty((0, dims))

    # generate feature vectors by concatenating multiframes
    vectors = np.zeros((n_vectors, dims))
    for t in range(n_frames):
        vectors[:, n_mels * t : n_mels * (t + 1)] = fbank[:, t : t + n_vectors].T
    '''
    return fbank

def file_list_to_singleastdata(file_name,
                         msg="calc...",
                         n_mels=128,
                         n_frames=1,
                         n_hop_frames=1,
                         n_fft=1024):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        data for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    #dims = n_mels * n_frames
    # iterate AST feature ()
    vectors_temp = []
    data = []
    vectors = file_to_AST(file_name,
                            n_mels=n_mels,
                            n_fft=n_fft)

    data = vectors
    data = data.reshape(1, n_fft, n_mels)
    return data


def file_list_to_astdata(file_list,
                         msg="calc...",
                         n_mels=128,
                         n_frames=1,
                         n_hop_frames=1,
                         n_fft=1024):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        data for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    #dims = n_mels * n_frames
    # iterate AST feature ()
    vectors_temp = []
    data = []
    for idx in tqdm(range(len(file_list)), desc=msg):
        vectors = file_to_AST(file_list[idx],
                                  n_mels=n_mels,
                                  n_fft=n_fft)

        if idx == 0:
            data = vectors
        else:
            data  = np.vstack((data, vectors))

    data = data.reshape(len(file_list), n_fft, n_mels)
    return data

def generator_ast_format(data):
    '''
    ndarry->torch
    B,T,F->B,C,T,F
    '''
    data = torch.from_numpy(data)
    data = data.unsqueeze(1)
    return data


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)

class Timer(object):
    """ Time code execution.

    Example usage::

        with Timer as t:
            sleep(10)
        print(t.secs)

    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.secs = 0
        self.msecs = 0
        self.start = 0
        self.end = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('elapsed time: %f ms' % self.msecs)




