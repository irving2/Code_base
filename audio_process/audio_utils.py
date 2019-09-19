#!/usr/bin/env python
# coding=utf-8
# Author : chenwen_hust@qq.com
# datetime:19-9-4 上午10:37
# project: snr_estimate


import os
import datetime

from pydub import AudioSegment
import array
from pydub.utils import get_array_type
from scipy.io import wavfile
import numpy as np
from subprocess import PIPE, run


# os.system('ffmpeg -i quite_pure_record.mp3 -acodec pcm_s16le -ac 1 -ar 16000 quite_pure_record.wav')     # 将mp3 转换为wav


def create_dirs(dirs:list):
    '''
    :param dirs: 需要创建的路径列表
    :return: exit_code  0:success  -1:failed
    '''
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print('创建路径失败:{}'.format(err))
        exit(-1)


def load_with_pydub(file_path):
    sound = AudioSegment.from_wav(file_path)
    return sound                            # pydub AudioSegment object


def get_array_from_pydub_obj(pydub_obj):
    bit_depth = pydub_obj.sample_width * 8      # 带宽  eg 2*8
    array_type = get_array_type(bit_depth)
    numeric_array = array.array(array_type, pydub_obj._data)
    return numeric_array


def load_auio(file_path):
    # sig, sr = librosa.load(file_path, sr, **kwargs)   # dtype 默认转换为[-1,1]的浮点数
    sr, sig = wavfile.read(file_path)
    if len(sig.shape) > 1:
        if sig.shape[1] == 1:
            sig = sig.squeeze()
        else:
            sig = sig.mean(axis=1)                      # multiple channels, average
    return sr, sig


def save_audio(path, sr, sig):
    wavfile.write(path, sr, sig.astype(np.int16))   # 16bit_PCM 直接保存float数据播放时有很大杂音。
    return None


def mixed_snr(sig, noise, snr=20):
    if noise.ndim > 1:         # handle stereo shape = (nsamples,2)
        noise = noise.mean(axis=1)
    assert len(noise) > len(sig), 'noise文件过短'

    if not np.isfinite(noise).all() or sum(noise)==0:
        raise ValueError('noise wav data has invalid Number')

    start_point = np.random.randint(len(noise)-len(sig))
    end_point = start_point+len(sig)
    noise = noise[start_point:end_point]

    sum_sig = np.sum(sig**2)
    sum_noise = np.sum(noise**2)
    alpha = np.sqrt(sum_sig/(sum_noise*pow(10, snr/10.0)))

    mixed_sig = sig+alpha*noise
    return mixed_sig


def split_wav(file_path, split_inter, save_dir, prefix_name=None):
    """
    :param Split_inter: 切割的时长,单位为秒
    :return: 分割出的文件数量
    """
    start_point = 0
    end_point = 0
    count = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not prefix_name:
        prefix_name = os.path.basename(save_dir)

    wav_data = AudioSegment.from_wav(file_path)
    while True:
        end_point = start_point + split_inter*1000
        if end_point > len(wav_data):
            break

        need_to_save = wav_data[start_point:end_point]
        save_name = '_'.join([prefix_name, str.zfill('%d' % count, 5), '.wav'])
        save_path = os.path.join(save_dir, save_name)
        need_to_save.export(save_path, format='wav')
        start_point = end_point
        count += 1
    return count

# def get_sox(filepath):
#     content = subprocess.check_output("soxi -D {}".format(filepath), shell=True).decode('utf-8')
#     return float(content)                      # 返回时间长度

def command_out(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result.stdout     # str类型

def cal_time(folder,audio_type='.wav'):
    def transform_time(sec):
        return datetime.timedelta(seconds=sec)  # days ,hours, minuntes, seconds

    def sec_to_hours(seconds):
        a = str(seconds // 3600)
        b = str((seconds % 3600) // 60)
        c = str((seconds % 3600) % 60)
        d = "{} hours {} mins {} seconds".format(a, b, c)
        return d

    if not os.path.exists(folder):
        raise FileNotFoundError
    total = 0
    for root, folders, files in os.walk(folder):
        for file in files:
            if file.endswith(audio_type.lower()):
                total += float(command_out("soxi -D {}".format(os.path.join(root, file))))
                print('total:', total)
    return sec_to_hours(total)


if __name__ == '__main__':
    ret = cal_time('/home/chenwen/snr_estimate/dataset/speak_num_wav')
    print(ret)