#!/usr/bin/env python
# coding=utf-8
# Author : chenwen_hust@qq.com
# datetime:19-9-3 下午2:50
# project: Code_base


import wave as _wave
import numpy as np
from scipy.io import wavfile
import librosa
from scipy.signal import spectrogram
from pydub import AudioSegment


filename = 'T0055G0013S0004.wav'
noise_file = 'T0055G7072S0215_mix_-8.95_dataset_noise_set_subway_00009.wav'

file_32 = 'new32.wav'

def _loadWAVWithWave(fileName):
    """ Load samples & sample rate from 24 bit WAV file """
    wav = _wave.open(fileName)
    rate = wav.getframerate()
    nchannels = wav.getnchannels()
    sampwidth = wav.getsampwidth()
    nframes = wav.getnframes()
    data = wav.readframes(nframes)
    wav.close()
    array = _wav2array(nchannels, sampwidth, data)

    return array, rate


def _wav2array(nchannels, sampwidth, data):
    """data must be the string containing the bytes from the wav file."""
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = np.empty((num_samples, nchannels, 4), dtype = np.uint8)
        raw_bytes = np.fromstring(data, dtype = np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        result = np.fromstring(data, dtype='<%s%d' % (dt_char, sampwidth))
        # result = result.reshape(-1, nchannels)
    return result                                                                    # int16 samplewidth=2 都是返回整数


normFact = {'int8': (2**7) -1,
            'int16': (2**15)-1,
            'int24': (2**23)-1,
            'int32': (2**31)-1,
            'int64': (2**63)-1,
            'float32': 1.0,
            'float64': 1.0}


if __name__ == '__main__':
    pydub_data = AudioSegment.from_wav(noise_file)
    # d1, rate1 = _loadWAVWithWave(noise_file)
    # rate2,d2 = wavfile.read(noise_file)
    # wavfile.write('new32.wav', rate2, d2.astype(np.int32))
    # # print(rate1,rate2)
    # # print(d1.shape,d2.shape)
    #
    #
    data, rate = librosa.load(noise_file, sr=None)
    # data = data.T
    # data = data*normFact['int' + str(16)]
    # data = np.int16(data)
    # # print(d2[:5])
    # # print(data[:5])
    # #
    # # print()
    #
    # f,t,sxx = spectrogram(data,rate)
    # f2,t2,sxx2 = spectrogram(d2,rate2)
    print()