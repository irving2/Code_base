from pdb import set_trace
import os
import subprocess
from tempfile import NamedTemporaryFile

from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data.sampler import Sampler
import scipy.io.wavfile as wav

import librosa
import numpy as np
import scipy.signal
import torch
# import torchaudio
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# import ipdb
    
windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}
from scipy.signal.windows import gaussian
from scipy.signal import convolve

def compute_mute_flag(s0):
    a = s0[:]
    a = a.astype('float32')/np.max(a)
    s = a**2*0.25
    w = gaussian(32,1)
    s = convolve(s,w)
   # plt.plot(s[:102400*5])
    s[0] = 0
    flag = np.array(s > 0.030,'uint8')
    return flag
def trim_array(s0,margin):
  #  set_trace()
    flag = compute_mute_flag(s0)
    start = -1
    for k in range(len(flag)):
        if flag[k] == 1:
            start = k
            break
       # if i == 0 and  start !=-1:
    end = -1
    for k in range(1,len(flag)):
        if flag[len(flag)-k] == 1:
            end = len(flag)-k
            break
           
  #  set_trace()    
    start = max(start-margin,0)
    
   # end = min(end+,s0.shape[0]-1)
    end = min(end+margin,len(flag))
    
    return s0[start:end]



def load_audio(path):
    # sound, _ = torchaudio.load(path, normalization=True)
    # sound = sound.numpy().T
    rate, sound = wav.read(path)
    sound = sound.astype('float32')
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound


class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError


class NoiseInjection(object):
    def __init__(self,
                 path=None,
                 sample_rate=16000,
                 noise_levels=(0, 0.5)):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """
        if not os.path.exists(path):
            print("Directory doesn't exist: {}".format(path))
            raise IOError
        self.paths = path is not None and librosa.util.find_files(path)
        self.sample_rate = sample_rate
        self.noise_levels = noise_levels

    def inject_noise(self, data):
        noise_path = np.random.choice(self.paths)
        noise_level = np.random.uniform(*self.noise_levels)
        return self.inject_noise_sample(data, noise_path, noise_level)

    def inject_noise_sample(self, data, noise_path, noise_level):
        noise_len = get_audio_length(noise_path)
        data_len = len(data) / self.sample_rate

        if noise_len < data_len:
            return data
        noise_start = np.random.rand() * (noise_len - data_len)
        noise_end = noise_start + data_len
        noise_dst = audio_with_sox(noise_path, self.sample_rate, noise_start, noise_end)
        assert len(data) == len(noise_dst)
        
#         ipdb.set_trace()
        noise_energy = np.sqrt(noise_dst.dot(noise_dst) / noise_dst.size)
        data = data.astype('int64')   # 不加的话，data_energy=nan
        data_energy = np.sqrt(data.dot(data) / data.size)
#         print (f'data energy: {data_energy}')
#         a = noise_level * noise_dst * data_energy / noise_energy
#         print(data_energy / noise_energy)
        data2 = data + noise_level * noise_dst * data_energy / noise_energy

        if not np.isfinite(data2).all():
            print('[info]:Noise injection failed',noise_path)
            data2=data

        # debug
        # invalid_num = [e for e in data2 if np.isfinite(e)==False]
        # if len(invalid_num)>0:
        #     # print(invalid_num,len(invalid_num),len(data2))
        #     print('data_energy',data_energy)
        #     print('noise_dst.any()', noise_dst.any())
        #     print('noise_dst.ratio', len(noise_dst[noise_dst==0]),len(noise_dst[noise_dst!=0]))
        #     print('noise_dst', max(noise_dst),min(noise_dst))
        #     print('noise_energy',noise_energy,noise_dst[:10])
        # # else:
        # #     print('normal:data_energy', data_energy)
        # #     print('normal:noise_energy', noise_energy)
        #
        # if not np.isfinite(data2).all():
        #     print('invalid value in data2-----------')
        #
        # if not np.isfinite(noise_dst).all():
        #     print('invalid value in noise_dst-----------')
        #
        # if not np.isfinite(noise_energy).all():
        #     print('invalid value in noise_energy-----------')
        #
        # if not np.isfinite(data_energy).all():
        #     print('invalid value in data_energy-----------')
        # debug

        return data2


class SpectrogramParser(AudioParser):
    def __init__(self, audio_conf, normalize=False, augment=False,trim=True):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        super(SpectrogramParser, self).__init__()
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.normalize = normalize
        self.augment = augment
        self.noiseInjector = NoiseInjection(audio_conf['noise_dir'], self.sample_rate,
                                            audio_conf['noise_levels']) if audio_conf.get(
            'noise_dir') is not None else None
        self.noise_prob = audio_conf.get('noise_prob')
        self.trim = trim

    def parse_audio(self, audio_path):
        if self.augment:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate)
            if self.trim:
                y = trim_array(y,np.random.randint(0,16000*0.5))
                
            
        else:
            y = load_audio(audio_path)
            
      #  set_trace()
        #lt.plot(y)
        #lt.show()
            if self.trim:
                #y = trim_array(y,0)
                y = trim_array(y,np.random.randint(0,16000*0.5))
            
        
       #plt.plot(y)
       #plt.show()
        if self.noiseInjector:
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                y = self.noiseInjector.inject_noise(y)
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)
        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError


class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, audio_conf, manifest_filepath, labels, normalize=False, augment=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...

        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        super(SpectrogramDataset, self).__init__(audio_conf, normalize, augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]
        spect = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript_path)

#         print (spect)
    
        # print (transcript)
        # [2, 6, 3, 6, 3, 8, 3, 11, 11, 8, 11, 2, 3, 4, 6, 4, 9, 10]
        return spect, transcript, audio_path

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        n = len(transcript)  
#         transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        t_list = [self.labels_map.get(x) for x in list(transcript)]
        transcript = list(filter(None, t_list))
#         set_trace()
#         print(t_list)
#         if len(transcript) != n:
#             print (transcript)
#             print(transcript_path)
        assert(len(transcript)==n)
        return transcript

    def __len__(self):
        return self.size


def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    filenames = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        fn = sample[2]
        filenames.append(fn)
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes, filenames


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)


class DistributedBucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1, num_replicas=None, rank=None):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(DistributedBucketingSampler, self).__init__(data_source)
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.data_source = data_source
        self.ids = list(range(0, len(data_source)))
        self.batch_size = batch_size
        self.bins = [self.ids[i:i + batch_size] for i in range(0, len(self.ids), batch_size)]
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.bins) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        offset = self.rank
        # add extra samples to make it evenly divisible
        bins = self.bins + self.bins[:(self.total_size - len(self.bins))]
        assert len(bins) == self.total_size
        samples = bins[offset::self.num_replicas]  # Get every Nth bin, starting from rank
        return iter(samples)

    def __len__(self):
        return self.num_samples

    def shuffle(self, epoch):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(epoch)
        bin_ids = list(torch.randperm(len(self.bins), generator=g))
        self.bins = [self.bins[i] for i in bin_ids]


def get_audio_length(path):
    output = subprocess.check_output(['soxi -D \"%s\"' % path.strip()], shell=True)
    return float(output)


def audio_with_sox(path, sample_rate, start_time, end_time):
    """
    crop and resample the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1".format(path, sample_rate,
                                                                                               tar_filename, start_time,
                                                                                               end_time)
        
        os.system(sox_params)
        y = load_audio(tar_filename)
        return y


def augment_audio_with_sox(path, sample_rate, tempo, gain):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                                      augmented_filename,
                                                                                      " ".join(sox_augment_params))
#         import ipdb
#         ipdb.set_trace()
#         print (sox_params)
        os.system(sox_params)
        y = load_audio(augmented_filename)
        return y


def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15),
                                  gain_range=(-6, 8)):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value)
    return audio