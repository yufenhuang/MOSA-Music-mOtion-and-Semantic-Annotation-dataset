
import os, sys, copy, csv
import scipy
import joblib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d
import datetime
import pandas as pd
import sklearn.preprocessing

from moviepy.editor import VideoFileClip, AudioFileClip
import librosa
import soundfile as sf 
import tensorflow as tf
from tensorflow import keras
from argparse import ArgumentParser


# show version info
print ("[info] Current Time:     " + datetime.datetime.now().strftime('%Y/%m/%d  %H:%M:%S'))
print ("[info] Python Version:   " + sys.version.split('\n')[0].split(' ')[0])
print ("[info] Working Dir:      " + os.getcwd()+'/')
print ("[info] Tensorflow:       " + tf.__version__)
print ("[info] Keras:            " + keras.__version__)


# limit GPU memory usage
print("[info] Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            print('set memory limit:', gpu)
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)



np.set_printoptions(suppress=True, precision=3)


def load_parameters():
    global data_sampling_rate, audio_normalize_range, motion_normalize_range, clip_len
    global body_marker_num, instrument_marker_num
    global beat_label_augment_time, downbeat_label_augment_time, phrase_label_augment_time
    
    clip_len = 16
    data_sampling_rate = 40
    audio_normalize_range = (0, 30)
    motion_normalize_range = 40
    body_marker_num = 30 # 30 for motion generation, 22 for time/expression semantics
    instrument_marker_num = 4 # 4 for motion generation, 0 for time/expression semantics
    beat_label_augment_time = 0.05 # beat label extend +- n seconds
    downbeat_label_augment_time = 0.05 # downbeat label extend +- n seconds
    phrase_label_augment_time = 0.5 # phrase label extend +- n seconds


def load_violin_joint_marker_list():

    global marker_name_list, new_joint_list, norm_ref_list

    marker_name_list = [
        "RFHD", "LFHD", "RBHD", "LBHD", "C7", "T10", "CLAV", "STRN",
        "RSHO", "RELB", "RWRA", "RWRB", "RFIN", "LSHO", "LELB", "LWRA", "LWRB", "LFIN",
        "RASI", "LASI", "RPSI", "LPSI", "RKNE", "RHEE", "RTOE", "RANK", "LKNE", "LHEE", "LTOE", "LANK",
        "VIOLINE", "VIOLINB", "BOWE", "BOWB"
        ]


    # combine markers to joint list
    new_joint_list = [
        ['Head', 0, 12], ['Neck', 12, 18], ['Root', 54, 66],
        ['Rshoulder', 24, 27], ['Relbow', 27, 30], ['Rwrist', 30, 36], ['Rfinger', 36, 39],
        ['Lshoulder', 39, 42], ['Lelbow', 42, 45], ['Lwrist', 45, 51], ['Lfinger', 51, 54],
        ['Rknee', 66, 69], ['Rhee', 69, 72], ['Rank', 75, 78], ['Rtoe', 72, 75], 
        ['Lknee', 78, 81], ['Lhee', 81, 84], ['Lank', 87, 90], ['Ltoe', 84, 87],
        ['Vtop', 90, 93], ['Vbom', 93, 96], ['Btop', 96, 99], ['Bbom', 99, 102]
        ]
   
    # the reference joints used for normalization
    norm_ref_list = [['Rtoe', 72, 75], ['Ltoe', 84, 87]]



def load_joint_list():
    global joint_list, joint_color_list, joint_line_list, ref_point
    # violin joints
    joint_list =  [
        'Head','Neck','Root', 
        'Rshoulder', 'Relbow','Rwrist','Rfinger',
        'Lshoulder','Lelbow', 'Lwrist', 'Lfinger',
        'Rknee','Rank','Rtoe', 
        'Lknee','Lank','Ltoe',
        'Vtop','Vbom','Btop', 'Bbom'
    ]

    ref_point = ['feet']

    joint_color_list = [
        'r', 'r', 'r',
        'b', 'b', 'b', 'b',
        'g', 'g', 'g', 'g',
        'cyan', 'cyan', 'cyan',
        'lime', 'lime', 'lime',
        'black', 'black', 'black', 'black'
    ]

    joint_line_list = [
        [0, 1], [1, 2], [1, 3], [1, 7], 
        [3, 4], [4, 5], [5, 6],
        [7, 8], [8, 9], [9, 10],
        [11, 12], [12, 13],
        [14, 15], [15, 16],
        [17, 18], [19, 20]
    ]



def load_recon_joint_list():

    global recon_joint_list, recon_joint_color_list, recon_joint_line_list

    recon_joint_list =  [
    'Head','Neck','Root', 
    'Rshoulder', 'Relbow','Rwrist','Rfinger',
    'Lshoulder','Lelbow', 'Lwrist', 'Lfinger',
    'Vtop','Vbom','Btop', 'Bbom']

    recon_joint_color_list = [
        'r', 'r', 'r',
        'b', 'b', 'b', 'b',
        'g', 'g', 'g', 'g',
        'black', 'black', 'black', 'black']

    recon_joint_line_list = [
        [0, 1], [1, 2], [1, 3], [1, 7], 
        [3, 4], [4, 5], [5, 6],
        [7, 8], [8, 9], [9, 10],
        [11, 12], [13, 14]]
    


def read_annot_csv(annot_file):

    with open(annot_file, newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        data_list = list(data)
        data_key = data_list[0]
        data_list = data_list[1:-1]
        
    return data_key, data_list


def get_note_annot_dict_from_folder(annot_folder, name_1, name_2, name_3):

    '''
    read annotation files from folder
    parameters:
    annot_folder: str, root directory of annot files
    name_1, name_2, name_3: str, the prefix of files

    returns: 
    note_annot_dict:note list of info in annotation files
    {'note_id', 'onset', 'offset', 'score_position', 'score_position_offset' 'note', 'midi_number', 'note_duration', 'bar_id', 'bar_position', 'beat', 'downbeat', 'phrase', 'dyn', 'articu'}

    beat_list: a list of beat time
    downbeat_list: a list of downbeat time
    phrase_list: a list of phrase onset time
    '''

    # define annot file names
    song_note_time_file = os.path.join(annot_folder, name_1 + '_' + name_2 + '_' + name_3 + '_align_notetime.csv')
    song_beat_file = os.path.join(annot_folder, name_1 + '_' + name_2 + '_' + name_3 + '_beat.csv')
    song_downbeat_file = os.path.join(annot_folder, name_1 + '_' + name_2 + '_' + name_3 + '_downbeat.csv')
    song_phrase_file = os.path.join(annot_folder, name_1 + '_' + name_2 + '_' + name_3 + '_phrase_section.csv')
    song_expression_file = os.path.join(annot_folder, name_1 + '_' + name_2 + '_' + name_3 + '_expression.csv')

    add_key_list = ['beat', 'downbeat', 'phrase', 'dyn', 'articu']
    initial_list = [0, 0, 0, 'none', 'none']
    
    # read note time file to dict
    note_key, note_list = read_annot_csv(song_note_time_file)
    # print('Read note len:', len(note_list))

    # get note time dict
    note_annot_dict = []
    annot_int_key_list = ['note_id', 'midi_number', 'bar_id']
    annot_float_key_list = ['onset', 'offset', 'score_position', 'score_position_offset', 'note_duration', 'bar_position']

    for note_inx in range(len(note_list)):
        note_item = {}

        # save note info to dict
        for key_inx in range(len(note_key)):
            key = note_key[key_inx]

            if key in annot_int_key_list:
                note_item[key] = int(note_list[note_inx][key_inx])
            elif key in annot_float_key_list:
                note_item[key] = float(note_list[note_inx][key_inx])
            else:
                note_item[key] = str(note_list[note_inx][key_inx])
        
        # initiate annot
        for add_key_inx in range(len(add_key_list)):
            add_key = add_key_list[add_key_inx]
            note_item[add_key] = initial_list[add_key_inx]

        note_annot_dict.append(note_item)
    
    print('Note dict len:', len(note_annot_dict))
    # print('Note dict:', note_annot_dict[:2])

    
    # read annot files
    _, beat_list = read_annot_csv(song_beat_file)
    _, downbeat_list = read_annot_csv(song_downbeat_file)
    _, phrase_list = read_annot_csv(song_phrase_file)
    express_key, express_list = read_annot_csv(song_expression_file)

    beat_list = [float(item[0]) for item in beat_list]
    downbeat_list = [float(item[0]) for item in downbeat_list]
    phrase_list = [float(item[0]) for item in phrase_list]

    # get express dict
    express_dict = []
    express_float_key_list = ['onset', 'offset']

    for annot_inx in range(len(express_list)):
        express_item = {}

        for key_inx in range(len(express_key)):
            key = express_key[key_inx]

            if key in express_float_key_list:
                express_item[key] = float(express_list[annot_inx][key_inx])
            else:
                express_item[key] = str(express_list[annot_inx][key_inx])
        
        express_dict.append(express_item)
    
    # print('Beat list:', beat_list)
    # print('Downbeat list:', downbeat_list)
    # print('Phrase list:', phrase_list)
    # print('Express dict:', express_dict)

    # save annot to dict
    for note_inx in range(len(note_annot_dict)):
        if note_annot_dict[note_inx]['score_position'] in beat_list:
            note_annot_dict[note_inx]['beat'] = 1
        if note_annot_dict[note_inx]['score_position'] in downbeat_list:
            note_annot_dict[note_inx]['downbeat'] = 1
        if note_annot_dict[note_inx]['score_position'] in phrase_list:
            note_annot_dict[note_inx]['phrase'] = 1
    
    # save express to dict
    dyn_type_list = ['ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff']
    arti_type_list = ['legato', 'staccato']

    for express_item in express_dict:

        # dyn annot
        for dyn_name in dyn_type_list:
            if express_item['expression'] == dyn_name:
                dyn_onset = express_item['onset']
                dyn_offset = express_item['offset']

                for note_inx in range(len(note_annot_dict)):
                    if dyn_onset <= note_annot_dict[note_inx]['score_position'] < dyn_offset:
                        note_annot_dict[note_inx]['dyn'] = dyn_name

        # arti annot
        for arti_name in arti_type_list:
            if express_item['expression'] == arti_name:
                arti_onset = express_item['onset']
                arti_offset = express_item['offset']

                for note_inx in range(len(note_annot_dict)):
                    if arti_onset <= note_annot_dict[note_inx]['score_position'] < arti_offset:
                        note_annot_dict[note_inx]['articu'] = arti_name
  
    return note_annot_dict, beat_list, downbeat_list, phrase_list


def convert_note_to_time_annot(
        note_annot_dict, 
        beat_list, 
        downbeat_list, 
        phrase_list,
        sampling_rate = 40,
        beat_aug_time = 0.05,
        downbeat_aug_time = 0.05,
        phrase_aug_time = 0.5,
        if_aug_time = True
        ):
    
    '''
    input: from function 'get_note_annot_dict_from_folder': 
    note_annot_dict, beat_list, downbeat_list, phrase_list
    define sampling_rate: frame per second
    beat_aug_time, downbeat_aug_time,  phrase_aug_time: for augment spares labels, 
    +- time in seconds, 0 = no augmentation
    '''

    # no augment time beat/downbeat/phrase label for test data
    if if_aug_time == False:
        beat_aug_time = sampling_rate/2
        downbeat_aug_time = sampling_rate/2
        phrase_aug_time = sampling_rate/2

    
    # get beat, downbeat, phrase performance time
    score_note_time_list = [note['score_position'] for note in note_annot_dict]
    perf_note_time_list = [note['onset'] for note in note_annot_dict]

    perf_beat_time_list = np.interp(beat_list, score_note_time_list, perf_note_time_list)
    perf_downbeat_time_list = np.interp(downbeat_list, score_note_time_list, perf_note_time_list)
    perf_phrase_time_list = np.interp(phrase_list, score_note_time_list, perf_note_time_list)
    # print('perf beat time:', perf_beat_time_list)

    # initiate time annot
    start_time = note_annot_dict[0]['onset']
    end_time = note_annot_dict[-1]['offset']
    time_list = np.arange(start_time, end_time, 1/sampling_rate)
    print('start time', start_time)
    print('end time:', end_time)

    time_annot_list = []
    for time in time_list:
        frame = {
            'time': time, 'beat': 0, 'downbeat': 0, 'phrase': 0, 'dyn': 'none', 'arti': 'none', 'pitch': 'none', 'midi': np.nan
            }

        # get frame dyn & arti label
        for note in note_annot_dict:
            if note['onset'] <= time < note['offset']:
                frame['dyn'] = note['dyn']
                frame['arti'] = note['articu']
            
        # get frame beat, downbeat, phrase label
        for beat_time in perf_beat_time_list:
            if beat_time - beat_aug_time <= time <= beat_time + beat_aug_time:
                frame['beat'] = 1
        
        for downbeat_time in perf_downbeat_time_list:
            if downbeat_time - downbeat_aug_time <= time <= downbeat_time + downbeat_aug_time:
                frame['downbeat'] = 1
        
        for phrase_time in perf_phrase_time_list:
            if phrase_time - phrase_aug_time <= time <= phrase_time + phrase_aug_time:
                frame['phrase'] = 1
        
        # get frame pitch
        for note in note_annot_dict:
            if note['onset'] <= time < note['offset']:
                frame['pitch'] = note['note']
                frame['midi'] = librosa.note_to_midi(note['note'])

        time_annot_list.append(frame)

    # print(time_annot_list[-5:])
    return time_annot_list



def get_motion_data(
        motion_file, 
        time_annot_list, 
        motion_original_sampling_rate = 120,
        final_sampling_rate = 40,
        body_marker_num = 22,
        instrument_marker_num = 0
        ):
    '''
    read motion data from .csv files
    downsample/upsample motion data from 'motion_original_sampling_rate' to 'final_sampling_rate'
    input: motion_folder, names
    time_annot_list from function 'convert_note_to_time_annot'
    motion_original_sampling_rate: frame per second
    final_sampling_rate: frame per second
    trim_marker_num: only get the data for n markers * 3 axis (xyz)
    '''

    # get motion data
    motion_marker, motion_data = read_annot_csv(motion_file)
    motion_data = np.array(motion_data).astype(float)
    body_motion_data = motion_data[:, :body_marker_num*3]
    instrument_motion_data = motion_data[:, -(instrument_marker_num*3):]

    if instrument_marker_num > 0:
        motion_data = np.concatenate((body_motion_data, instrument_motion_data), axis=1)
    elif instrument_marker_num == 0:
        motion_data = body_motion_data

    motion_data = remove_data_nan(motion_data)
    motion_resample = resample_2d_array(
        motion_data, 
        motion_original_sampling_rate, 
        final_sampling_rate
        )

    # downsample_ratio = int(motion_original_sampling_rate/final_sampling_rate)
    # motion_resample = motion_data[::downsample_ratio, :]
    print('motion data original:', motion_data.shape)
    print('motion data resample:', motion_resample.shape)

    # clip motion data
    start_frame = int(time_annot_list[0]['time'] * final_sampling_rate)
    end_frame = int(time_annot_list[-1]['time'] * final_sampling_rate)
    frame_len = len(time_annot_list)
    motion_clip = motion_resample[start_frame: int(start_frame+frame_len), :]
    print('start frame, end frame:', start_frame, end_frame)
    print('motion data clip:', motion_clip.shape)

    return motion_clip


def get_audio_data(
        audio_file,
        time_annot_list,
        audio_sr = 22050,
        stft_hop_length = 512,
        final_sampling_rate = 40
        ):
    
    # get song spectrogram
    audio, sr = librosa.load(audio_file, sr=audio_sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000, hop_length=stft_hop_length)
    mel = librosa.power_to_db(mel, ref=np.max).T
    mel = remove_data_nan(mel)
    mel_time = librosa.times_like(mel, axis = 0)
    # print('mel time:', mel_time)

    # downsample
    audio_original_sample_rate = 1 / (stft_hop_length/audio_sr)
    audio_resample = resample_2d_array(
        mel, 
        audio_original_sample_rate, 
        final_sampling_rate
        )
    
    audio_time_resample = np.linspace(0, mel_time[-1], audio_resample.shape[0])
    print('audio data original:', mel.shape)
    print('audio data resample:', audio_resample.shape)
    # print('mel time:', mel_time.shape, mel_time[:10], mel_time[-10:])
    # print('resample time:', audio_time_resample.shape, audio_time_resample[:10], audio_time_resample[-10:])

    # clip audio data
    start_time = time_annot_list[0]['time']
    frame_len = len(time_annot_list)
    time_diff = np.absolute(audio_time_resample - start_time)
    start_inx = time_diff.argmin()
    end_inx = start_inx + frame_len
    audio_clip = audio_resample[start_inx:end_inx, :]
    # print('time diff:', time_diff.shape, time_diff[:10], time_diff[-10:])
    # print('start inx:', start_inx, end_inx)
    print('audio data clip:', audio_clip.shape)

    return audio_clip, start_time


def get_audio_feature(
        audio_file,
        time_annot_list,
        audio_sr = 22050,
        stft_hop_length = 512,
        final_sampling_rate = 40
        ):
    
    # define normalize scale
    norm_scaler = sklearn.preprocessing.MinMaxScaler(feature_range = (0, 10))

    # get spectral_flux
    audio, sr = librosa.load(audio_file, sr=audio_sr)
    spectral_flux = librosa.onset.onset_strength(y=audio, sr=sr)
    spectral_flux = norm_scaler.fit_transform(spectral_flux.reshape(-1, 1))
    # spectral_flux = np.squeeze(spectral_flux, axis = -1)
    # print('original spectral flux:', spectral_flux.shape)

    # get rms
    rms = librosa.feature.rms(y=audio)
    rms = np.squeeze(rms, axis=0)
    rms = norm_scaler.fit_transform(rms.reshape(-1, 1)) # normalize
    # rms = np.squeeze(rms, axis = -1)
    # print('original rms:', rms.shape)


    # get feature time
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000, hop_length=stft_hop_length)
    mel = librosa.power_to_db(mel, ref=np.max).T
    mel_time = librosa.times_like(mel, axis = 0)
    # print('original mel:', mel.shape)
    # print('mel time:', mel_time)

    # downsample
    audio_original_sample_rate = 1 / (stft_hop_length/audio_sr)
    spectral_flux_resample = resample_2d_array(
        spectral_flux, 
        audio_original_sample_rate, 
        final_sampling_rate
        )
    
    rms_resample = resample_2d_array(
        rms, 
        audio_original_sample_rate, 
        final_sampling_rate
        )
    
    # print('resample spectral flux:', spectral_flux_resample.shape)
    # print('resample rms flux:', rms_resample.shape)
    
    feature_time_resample = np.linspace(0, mel_time[-1], spectral_flux_resample.shape[0])
    # print('mel time:', mel_time.shape, mel_time[:10], mel_time[-10:])
    # print('resample time:', audio_time_resample.shape, audio_time_resample[:10], audio_time_resample[-10:])

    # clip audio data
    start_time = time_annot_list[0]['time']
    frame_len = len(time_annot_list)
    time_diff = np.absolute(feature_time_resample - start_time)
    start_inx = time_diff.argmin()
    end_inx = start_inx + frame_len
    flux_clip = spectral_flux_resample[start_inx:end_inx, :]
    rms_clip = rms_resample[start_inx:end_inx, :]
    # print('time diff:', time_diff.shape, time_diff[:10], time_diff[-10:])
    # print('start inx:', start_inx, end_inx)
  
    print('audio feature clip:', flux_clip.shape, rms_clip.shape)

    return flux_clip, rms_clip



def rescale_audio_data(data_arr, rescale_range=(0, 10)):
    '''
    input: data array
    rescale_range: (lower_bound, upper_bound)
    '''
    scaler = sklearn.preprocessing.MinMaxScaler(rescale_range)
    new_data_arr = scaler.fit_transform(data_arr)
    print('rescale data to range:', rescale_range)

    return new_data_arr


def remove_data_nan(data_arr, axis=0):
    data_arr = pd.DataFrame(data_arr)
    data_arr = data_arr.interpolate(axis= axis, limit_direction= 'both')
    data_arr = np.array(data_arr)

    return data_arr


def resample_2d_array(
        data_arr, 
        original_sample_rate, 
        final_sample_rate
        ):
    
    '''
    resample time (axis 0) for the ratio for the resample rate
    input: 2d data array [time, feature]
    output: resample 2d array [resample_time, feature]
    resample_time = time * final_sample_rate /original_sample_rate
    '''

    resample_num = int(data_arr.shape[0] * final_sample_rate / original_sample_rate)
    resample_arr = np.zeros((resample_num, data_arr.shape[1]))

    original_time = np.linspace(0, data_arr.shape[0], data_arr.shape[0])
    resample_time = np.linspace(0, data_arr.shape[0], resample_num)

    for marker_inx in range(data_arr.shape[1]):
        data_slice = data_arr[:, marker_inx]
        interp_fn = scipy.interpolate.interp1d(original_time, data_slice)
        resample_slice = interp_fn(resample_time)
        resample_arr[:, marker_inx] = resample_slice

    return resample_arr


def convert_str_to_class_label(
        str_label_arr,
        label,
        class_name):

    class_label_arr = np.zeros(str_label_arr.shape).astype(float)

    for inx in range(len(label)):
        item_inx = np.where(str_label_arr == label[inx])
        class_label_arr[item_inx] = class_name[inx]
    
    return class_label_arr


def remove_extreme_values(data_arr, threshold = (-100, 100)):
    '''
    remove extreme values
    input: data array
    threshold: (lower_bound, upper_bound)
    '''
    
    clean_arr = np.where(data_arr > threshold[1], threshold[1], data_arr)
    clean_arr = np.where(clean_arr < threshold[0], threshold[0], clean_arr)
    
    # count removed values
    remove_small = np.count_nonzero(data_arr < threshold[0])
    remove_large = np.count_nonzero(data_arr > threshold[1])
    # print('remove extreme large values:', remove_large, 'remove extreme small values:', remove_small)
    
    return clean_arr


def get_motion_velocity(data_arr, motion_sampling_rate = 120):

    '''
    input: data_arr: motion position array [time, feature]
    output: motion velocity array (cm per second) [time, feature]
    '''

    data_arr = data_arr.astype(float)
    data_arr = remove_data_nan(data_arr)
    data_roll = np.roll(data_arr, 1, axis=0)
    arr_diff = (data_arr - data_roll) * motion_sampling_rate / 10 # convert mm to cm
    arr_diff = arr_diff[1:, :]
    last = arr_diff[-1, :]
    last = last[np.newaxis, ...]
    arr_diff = np.concatenate((arr_diff, last), axis=0)

    return arr_diff


# get joint center position of the input motion segment data
def get_joint_center(seg_data):

    seg_x = np.mean(seg_data[:, ::3], axis=1)
    seg_y = np.mean(seg_data[:, 1::3], axis=1)
    seg_z = np.mean(seg_data[:, 2::3], axis=1)
    
    seg_center = np.array([seg_x, seg_y, seg_z]).T
    
    return seg_center


# translate motion_arr to local coordinate with ref_pos is (0, 0, 0) 
def get_local_coordinate(motion_arr, ref_pos):

    new_arr = np.zeros(motion_arr.shape)
    motion_arr = np.array(motion_arr)

    pos_x = motion_arr[:, ::3]
    pos_y = motion_arr[:, 1::3]
    pos_z = motion_arr[:, 2::3]

    trans_x = pos_x - ref_pos[0]
    trans_y = pos_y - ref_pos[1]
    trans_z = pos_z - ref_pos[2]

    new_arr[:, ::3] = trans_x
    new_arr[:, 1::3] = trans_y
    new_arr[:, 2::3] = trans_z
    
    return new_arr


# combine data from multiple joints to one single array
def combine_joints(comb_list, data_arr):
    '''
    combine several markers into a joint
    input: comb_list ['marker_name', marker_start_column, marker_end_column]

    '''
    comb_num = len(comb_list)
    comb_arr = np.zeros((data_arr.shape[0], comb_num*3))
    
    for comb_inx in range(comb_num):
        joint_name = comb_list[comb_inx][0]
        col_start = comb_list[comb_inx][1]
        col_end = comb_list[comb_inx][2]
        comb_arr[:, comb_inx*3: comb_inx*3+3] = data_arr[:, col_start:col_end]
        # print('combine joint:', joint_name, col_start, col_end)
    
    # print(comb_arr[0, :])
    # print(comb_arr[-1, :])
    new_joint_arr = get_joint_center(comb_arr)
    
    return new_joint_arr


def get_joint_arr_from_marker_arr(marker_arr):

        # calculate ref position
        ref_data = combine_joints(norm_ref_list, marker_arr) # combine Rtoe and Ltoe as the ref point
        ref_mean_pos = np.mean(ref_data, axis=0)
        print('ref position:', ref_mean_pos)

        # translate feet to the center
        norm_data = get_local_coordinate(marker_arr, ref_mean_pos)
        print('translation motion data:', norm_data.shape)

        # from 43 markers to 23 joints
        joint_arr = conver_marker_to_joint(marker_arr, new_joint_list)

        return joint_arr


def conver_marker_to_joint(marker_arr, new_joint_list):
    '''
    convert 43 markers to 23 joints
    input: marker_arr [time, markers*3 (43*3)]
    new_joint_list: [joint_name, marker_start_column, marker_end_column]
    output: joint_arr [time, joints*3 (23*3)]
    '''

    # get data for 23 joints
    joint_num = len(new_joint_list)
    temp_joint_arr = np.zeros((marker_arr.shape[0], joint_num*3))

    for joint_inx in range(joint_num):
        joint_name = new_joint_list[joint_inx][0]
        col_start = new_joint_list[joint_inx][1]
        col_end = new_joint_list[joint_inx][2]
        joint_center_data = get_joint_center(marker_arr[:, col_start:col_end])
        temp_joint_arr[:,joint_inx*3:joint_inx*3+3] = joint_center_data
        # print('get data:', joint_name, col_start, col_end, joint_center_data.shape)
        # print('write data:', joint_inx*3, joint_inx*3+3)

    # print('temp joint shape:', temp_joint_arr.shape)

    # combine ankle joints
    rank_arr = get_joint_center(temp_joint_arr[:, 36:42])
    lank_arr = get_joint_center(temp_joint_arr[:, 48:54])

    joint_arr = np.concatenate((temp_joint_arr[:, 0:36], 
                                rank_arr,
                                temp_joint_arr[:, 42:48],
                                lank_arr,
                                temp_joint_arr[:, 54:]), axis=1)

    print('joint shape:', joint_arr.shape)

    return joint_arr


def get_body_seg_data_from_joint_arr_2d(joint_arr):

    rhand = joint_arr[:, 12:21]
    lhand = joint_arr[:, 24:33]
    btop = joint_arr[:, 57:60]
    vtop = joint_arr[:, 51:54]

    pos_head = np.array(joint_arr[:, :9])
    pos_rhand = np.concatenate((btop, rhand), axis=-1)
    pos_lhand = np.concatenate((vtop, lhand), axis=-1)

    return pos_head, pos_rhand, pos_lhand


def shift_pos_to_joint_mean_2d(motion_arr):
    '''
    shift each joint to its own mean position
    get the mean pos of each joint's xyz
    input: 2d motion_arr [time, feature (joint*xyz)]
    output: shifted joint arr pos_shift [time, feature (joint*xyz)]
    the mean position of each joint pos_mean [feature (joint*xyz)]
    '''

    pos_mean = np.mean(motion_arr, axis= 0)
    pos_shift = np.array(motion_arr - pos_mean)

    return pos_shift, pos_mean


def clean_outlier_keep_dim_2d(data_arr, tolerance = 1.5, keep_dim = 1):
    '''
    clean outliers for 3d input arr
    keep one dimension
    
    input: data_arr[time, feature]
    
    output:clean_arr [time, feature]
    
    for motion data: keep_dim = -1 (feature, xyz)
    tolerance: Q75/Q25  +- Quartile*tolerance
    '''

    all_dim = [0, 1]
    clean_dim = tuple(set(all_dim) - set([keep_dim]))
    clean_arr = copy.deepcopy(data_arr)

    # get outlier upper/lower bound
    q75,q25 = np.percentile(data_arr,[75,25], axis=clean_dim)
    intr_qr = q75-q25
    upper_bound = q75+(tolerance*intr_qr) # [feature,]
    lower_bound = q25-(tolerance*intr_qr) # [feature,]

    # loop over feature dimension
    for f_inx in range(data_arr.shape[-1]):
        f_upper = upper_bound[f_inx] # value
        f_lower = lower_bound[f_inx] # value

        f_arr = data_arr[:, f_inx]
        f_arr = np.where(f_arr > f_upper, f_upper, f_arr)
        f_arr = np.where(f_arr < f_lower, f_lower, f_arr)
        clean_arr[:, f_inx] = f_arr
        clean_arr = np.array(clean_arr)
    
    return clean_arr


def normalize_data_all_dim(data_arr, norm_max = 1):
    '''
    normalize 2d or 3D input arr to the range [0, norm_max]
    
    input: data_arr [time, feature] or [batch, time, feature]
    
    output:norm_data [time, feature] or [batch, time, feature]
    data_min
    data_max
    
    '''
                                                    
    data_min = np.min(data_arr)
    data_max = np.max(data_arr)
    
    norm_data = np.array((data_arr - data_min) * norm_max / (data_max - data_min))
    
    return norm_data, data_min, data_max






def get_song_data_label_snippet_list(
        folder_dir,
        song_name,
        perf_name,
        trial_name,
        # snippet_hop_size = 2,
        audio_range = (0, 10),
        velocity_range = (-50, 50),
        # if_snippet_hop = True,
        if_aug_time = True,
        if_velocity_clean = True,
        if_audio_norm = True,
        ):
    

    # read annot files
    song_annot_csv_folder = os.path.join(folder_dir, song_name, perf_name, trial_name, "annotation", "annotations")

    song_note_annot_dict, song_beat_list, song_downbeat_list, song_phrase_list = get_note_annot_dict_from_folder(
        song_annot_csv_folder, 
        song_name, 
        perf_name, 
        trial_name
        )

    # for item in song_note_annot_dict:
    #     print(item)

    # get annot list
    

    song_annot_list = convert_note_to_time_annot(
        song_note_annot_dict, 
        song_beat_list, 
        song_downbeat_list, 
        song_phrase_list,
        sampling_rate = data_sampling_rate,
        beat_aug_time = beat_label_augment_time,
        downbeat_aug_time = downbeat_label_augment_time,
        phrase_aug_time = phrase_label_augment_time,
        if_aug_time = True
        )
        
    # print('beat list', song_beat_list)
    # print('downbeat list', song_downbeat_list)
    # print('phrase list', song_phrase_list)
    print('annot list len:', len(song_annot_list))

    # get motion data
    load_violin_joint_marker_list()
    song_motion_csv_folder = os.path.join(folder_dir, song_name, perf_name, trial_name)
    song_motion_file = os.path.join(song_motion_csv_folder, song_name + '_' + perf_name + '_' + trial_name + '_motion_norm.csv')

    song_motion_data = get_motion_data(
        song_motion_file, 
        song_annot_list, 
        motion_original_sampling_rate = 120,
        final_sampling_rate = data_sampling_rate,
        body_marker_num = body_marker_num,
        instrument_marker_num = instrument_marker_num
        )
    
    song_joint_data = get_joint_arr_from_marker_arr(song_motion_data)
    song_joint_mean = np.mean(song_joint_data, axis = 0)
    
    # check_nan(song_joint_data)
    # print('motion data')
    # for inx in range(3):
    #     print(inx, song_joint_data[inx, :])
    
    
    # get motion velocity
    song_velocity_data = get_motion_velocity(song_joint_data)
    # check_nan(song_velocity_data)
    # print('velocity data')
    # for inx in range(3):
    #     print(inx, song_velocity_data[inx, :])
    
    # remove extreme value
    if if_velocity_clean == True:
        song_velocity_data = remove_extreme_values(song_velocity_data, threshold = velocity_range)
        print('Remove outliers in motion velocity')
    
    # get audio data
    song_audio_csv_folder = os.path.join(folder_dir, song_name, perf_name, trial_name)
    song_audio_file = os.path.join(song_audio_csv_folder, song_name + '_' + perf_name + '_' + trial_name + '_audio.wav')

    song_audio_data, audio_start_time = get_audio_data(
        song_audio_file,
        song_annot_list,
        audio_sr = 22050,
        stft_hop_length = 512,
        final_sampling_rate = data_sampling_rate
        )
    
    # get audio feature
    song_flux_data, song_rms_data = get_audio_feature(
        song_audio_file,
        song_annot_list,
        audio_sr = 22050,
        stft_hop_length = 512,
        final_sampling_rate = data_sampling_rate
        )
   
    
    if if_audio_norm == True:
        song_audio_data = rescale_audio_data(song_audio_data, rescale_range = audio_range)
    
    # get data for 16 seconds
    win_start = 0
    win_end = data_sampling_rate*clip_len
    audio_seg = song_audio_data[win_start:win_end, :]
    position_seg = song_joint_data[win_start:win_end, :]
    flux_seg = song_flux_data[win_start:win_end, :]
    rms_seg = song_rms_data[win_start:win_end, :]
    label_seg = song_annot_list[win_start:win_end]

    # get annot
    beat_seg = np.array([item['beat'] for item in label_seg])
    downbeat_seg = np.array([item['downbeat'] for item in label_seg])
    phrase_seg = np.array([item['phrase'] for item in label_seg])
    dyn_seg = np.array([item['dyn'] for item in label_seg])
    arti_seg = np.array([item['arti'] for item in label_seg])
    midi_seg = np.array([item['midi'] for item in label_seg])
    midi_seg =  remove_data_nan(midi_seg, axis=0)
        
    # # get snippet
    # song_snippet_list = get_snippet(
    #     audio_data = song_audio_data, 
    #     position_data = song_joint_data,
    #     velocity_data = song_velocity_data,
    #     flux_data = song_flux_data,
    #     rms_data = song_rms_data,
    #     label_list = song_annot_list,
    #     id_name = song_name + '_' + perf_name + '_' + trial_name,
    #     snippet_time_window = snippet_time_window,
    #     sampling_rate = data_sampling_rate,
    #     snippet_hop_size = snippet_hop_size,
    #     if_snippet_hop = if_snippet_hop)

    inference_pkg = {
        'audio_data': audio_seg,
        'motion_pos_data': position_seg,
        'motion_pos_mean': song_joint_mean,
        'beat_label': beat_seg,
        'downbeat_label':downbeat_seg,
        'phrase_label': phrase_seg,
        'dyn_label': dyn_seg,
        'arti_label': arti_seg,
        'midi_label': midi_seg,
        'flux': flux_seg,
        'rms': rms_seg,
        'audio_start_time': np.array(audio_start_time)
        }

    for key in inference_pkg:
        try:
            print(key, inference_pkg[key].shape)
        except:
            print(key, len(inference_pkg[key]))
    
    return inference_pkg



def get_test_label_from_pkg(data_pkg):
    '''
    reorganize data and label as array for input data package
    input: data_pkg: list {keys: array}
    output: data_pkg dict {keys, array}
    '''

    # # shuffle data & label
    # data_pkg = random.sample(data_pkg, len(data_pkg))

    # get data & label array
    load_data_pkg = {}

    print('=====')
    print('Get data:')

    for key in data_pkg.keys():
        
        data_arr = np.array(data_pkg[key])
        
        # try:
        #     data_arr = np.array([item[key] for item in data_pkg], dtype=np.float32)
        
        # except: 
        #     data_arr = np.array([item[key] for item in data_pkg])
        
        # clean up data
        if key == 'audio_data' or key == 'motion_pos_data' or key == 'motion_vel_data':
            new_data_arr = remove_data_nan(data_arr, axis=1)
            # print(key, 'processing audio, motion data')
            
        # convert string label to class label
        if key == 'dyn_label':

            new_data_arr = convert_str_to_class_label(
                data_arr,
                label = ['ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff'],
                class_name = [0,0,1,2,3,4,5,5])
            new_data_arr = new_data_arr[..., np.newaxis]
            # print(key, 'processing dyn label')
        
        elif key == 'arti_label':

            new_data_arr = convert_str_to_class_label(
            data_arr,
            label = ['legato', 'none', 'staccato'],
            class_name = [0,1,2])
            new_data_arr = new_data_arr[..., np.newaxis]
            # print(key, 'processing arti label')
        
        elif key == 'beat_label' or key == 'downbeat_label' or key== 'phrase_label':
            new_data_arr = data_arr[..., np.newaxis]
            # print(key, 'processing beat, downbeat, phrase labels')
        
        else:
            new_data_arr = data_arr
            # print(key, 'maintain original data')
        
        load_data_pkg[key] = new_data_arr

        # check data values
    print('---') 
    print('data values:')
    for key in load_data_pkg.keys():
        print(key, load_data_pkg[key].shape)
        print(key, 'max: {:.2f}, min: {:.2f}, mean: {:.2f}'.format(
                np.nanmax(load_data_pkg[key]), 
                np.nanmin(load_data_pkg[key]),
                np.nanmean(load_data_pkg[key])))

    return load_data_pkg


def get_test_data_from_pkg(data_pkg):
    '''
    combine label features
    normalize motion position, shift to the mean pos as 0
    divide motion position data into head, rhand, lhand segments
    input: data_pkg from fn 'get_test_label_from_pkg'
    output: data_pkg {
    comb_label, pos_rhand, pos_lhand, pos_head
    }
    '''

    new_data_pkg = {}

    # combine labels
    not_num_arr = np.ones(data_pkg['beat_label'].shape)
    comb_label = np.concatenate((
        not_num_arr, 
        data_pkg['beat_label'],
        data_pkg['downbeat_label'],
        data_pkg['phrase_label'],
        data_pkg['dyn_label'],
        data_pkg['arti_label'],
        data_pkg['midi_label'],
        data_pkg['midi_label'],
        data_pkg['flux'],
        data_pkg['rms']), axis = -1)
    
    new_data_pkg['comb_label'] = comb_label

    # get audio data
    new_data_pkg['audio_data'] = data_pkg['audio_data']
    new_data_pkg['audio_start_time'] = data_pkg['audio_start_time']

    new_data_pkg['motion_pos_mean_all'] = data_pkg['motion_pos_mean']

    # get body segment data
    pos_head_data, pos_rhand_data, pos_lhand_data = get_body_seg_data_from_joint_arr_2d(data_pkg['motion_pos_data'])
    new_data_pkg['motion_pos_head'] =pos_head_data
    new_data_pkg['motion_pos_rhand'] =pos_rhand_data
    new_data_pkg['motion_pos_lhand'] =pos_lhand_data

    # normalize motion
    motion_key_list = ['motion_pos_head', 'motion_pos_rhand', 'motion_pos_lhand']
    mean_key_list = ['mean_pos_head', 'mean_pos_rhand', 'mean_pos_lhand']
    min_max_key_list = ['min_max_head', 'min_max_rhand', 'min_max_lhand']

    for inx in range(len(motion_key_list)):
        # shift to the pos where joint mean is 0
        motion_key = motion_key_list[inx]
        new_data_pkg[motion_key], new_data_pkg[mean_key_list[inx]] = shift_pos_to_joint_mean_2d(new_data_pkg[motion_key])
        new_data_pkg[motion_key] = clean_outlier_keep_dim_2d(
            new_data_pkg[motion_key], 
            tolerance = 1.5, keep_dim = 1)
        
        # normalize the range of motion data
        new_data_pkg[motion_key], data_min, data_max = normalize_data_all_dim(
            new_data_pkg[motion_key], norm_max = motion_normalize_range)
        new_data_pkg[min_max_key_list[inx]] = np.array([data_min, data_max])

    # check data values
    print('---') 
    print('input data:')
    for key in new_data_pkg.keys():
        print(key, new_data_pkg[key].shape)
        print(key, 'max: {:.2f}, min: {:.2f}, mean: {:.2f}'.format(
                np.nanmax(new_data_pkg[key]), 
                np.nanmin(new_data_pkg[key]),
                np.nanmean(new_data_pkg[key])))
        # print('data type:', load_data_pkg[key].dtype)

        # if key == 'audio_data' or key == 'motion_pos_data' or key == 'motion_vel_data':
        #     print(key, 'max: {:.2f}, min: {:.2f}, mean: {:.2f}'.format(
        #         np.nanmax(load_data_pkg[key]), 
        #         np.nanmin(load_data_pkg[key]),
        #         np.nanmean(load_data_pkg[key])))
        
    del data_pkg
    return new_data_pkg



# define loss functions

def mse_2d_fn(labels, preds):
    '''
    calcuate mse for flatten 2d array
    https://www.tensorflow.org/api_docs/python/tf/keras/metrics/mean_squared_error
    '''
    
    epsilon=1e-7
    
    labels = tf.reshape(labels, [-1])
    preds = tf.reshape(preds + epsilon, [-1])
    
    mse = tf.keras.losses.mean_squared_error(labels, preds)

    return mse


def mae_2d_fn(labels, preds):

    epsilon=1e-7
    preds = preds + epsilon
    
#     labels = tf.reshape(labels, [-1])
#     preds = tf.reshape(preds + epsilon, [-1])

#     mae = tf.keras.losses.MeanAbsoluteError(labels, preds)
    mae = tf.abs(labels - preds) / tf.cast(tf.size(labels), tf.float32)

    return mae


def time_wise_mae_fn(labels, preds):
    '''
    calculate pairwise mae along time axis for each dimensions (x, y, z)
    from https://www.tensorflow.org/ranking/api_docs/python/tfr/keras/losses/PairwiseMSELoss
    tfr.keras.losses.PairwiseMSELoss()
    '''
    
    epsilon=1e-7
    preds = preds + epsilon
    
    labels_tf = tf.transpose(labels, [0, 2, 1]) # [b, 3, t]
    preds_tf = tf.transpose(preds, [0, 2, 1]) # [b, 3, t]
    label_diff = labels_tf[:, :, :, None] - labels_tf[:, :, None, :] # [b, 3, t, t]
    preds_diff = preds_tf[:, :, :, None] - preds_tf[:, :, None, :] # [b, 3, t, t]
    pairwise_loss = (preds_diff - label_diff)**2 # [b, 3, t, t]
    pairwise_loss = tf.reduce_mean(pairwise_loss)
    
    return pairwise_loss
    

def dim_wise_mae_fn(labels, preds):
    
    epsilon=1e-7
    preds = preds + epsilon
    
    # labels_tf = tf.transpose(labels, [0, 2, 1]) # [b, 3, t]
    # preds_tf = tf.transpose(preds, [0, 2, 1]) # [b, 3, t]
    # label_diff = labels_tf[:, :, :, None] - labels_tf[:, :, None, :] # [b, t, 3, 3]
    # preds_diff = preds_tf[:, :, :, None] - preds_tf[:, :, None, :] # [b, t, 3, 3]
    label_diff = labels[:, :, :, None] - labels[:, :, None, :] # [b, t, 3, 3]
    preds_diff = preds[:, :, :, None] - preds[:, :, None, :] # [b, t, 3, 3]
    dim_loss = (preds_diff - label_diff)**2 # [b, t, 3, 3]
    dim_loss = tf.reduce_mean(dim_loss)
    
    return dim_loss    
    
    

def custom_mse_loss_fn(labels, preds):
    mse_loss_weight = 1
    mae_loss_weight = 0
    time_loss_weight = 2
    dim_loss_weight = 2
    
    mse_weight = mse_loss_weight
    mae_weight = mae_loss_weight
    time_weight = time_loss_weight
    dim_weight = dim_loss_weight
    
    mse_loss = mse_2d_fn(labels, preds)
    mae_loss = mae_2d_fn(labels, preds)
    time_loss = time_wise_mae_fn(labels, preds)
    dim_loss = dim_wise_mae_fn(labels, preds)
    
    total_loss = (mse_weight*mse_loss) + (mae_weight*mae_loss) + (time_weight*time_loss) + (dim_weight*dim_loss)
    # print('mse_loss:', mse_loss, 'pairwise_loss:', pairwise_loss)
    
    return total_loss


def get_prediction(model_name, test_data):
    '''
    get pred_pkg
    input: model_name (include the full directory)
    test_data['audio_data'] ['comb_data]
    
    output: pred_pkg {'pred_pos_rhand', 'pred_pos_lhand', 'pred_pos_head',
    'truth_pos_rhand', 'truth_pos_lhand', 'truth_pos_head',
    'mean_pos_rhand', 'mean_pos_lhand', 'mean_pos_head',
    'min_max_rhand', 'min_max_lhand', 'min_max_head', 
    'motion_pos_mean_all', 'audio_start_time'}
    '''

    # load model
    model = tf.keras.models.load_model(
        model_name, 
        custom_objects={"custom_mse_loss_fn": custom_mse_loss_fn})

    model.summary()

    # get prediction
    audio_data = test_data['audio_data'][np.newaxis,...] # [1, time, feature]
    feature_data = test_data['comb_label'][np.newaxis,...] # [1, time feature]

    prediction = model.predict([audio_data, feature_data])

    pred_pkg = {
        'pred_pos_rhand': np.squeeze(prediction[0], axis = 0), # [time, feature]
        'pred_pos_lhand': np.squeeze(prediction[1], axis = 0),
        'pred_pos_head': np.squeeze(prediction[2], axis = 0),
    }

    # get ground truth data
    test_data_key_list = ['motion_pos_rhand', 'motion_pos_lhand', 'motion_pos_head']
    pred_pkg_key_list = ['truth_pos_rhand', 'truth_pos_lhand', 'truth_pos_head']
    
    for inx in range(len(test_data_key_list)):
        pred_pkg[pred_pkg_key_list[inx]] = test_data[test_data_key_list[inx]]
    
    # get pos mean, min, & max
    append_key_list = [
        'mean_pos_rhand', 'mean_pos_lhand', 'mean_pos_head',
        'min_max_rhand', 'min_max_lhand', 'min_max_head', 
        'motion_pos_mean_all', 'audio_start_time']
    
    for key in append_key_list:
        pred_pkg[key] = test_data[key]
    
    # check result pkg
    for key in pred_pkg:
        print(key, pred_pkg[key].shape)
    
    return pred_pkg


def transform_to_original_scale(norm_arr, max_value, min_value, norm_max = 1):
    '''
    transform normalized motion arr to the original position
    norm_arr = (original_arr - data_min_arr) * norm_max / (data_max_arr - data_min_arr)
    input: norm_motion_arr [time, feature]
    max_arr [feature]
    min_arr [feature]
    norm_max: the max value in norm_motion_arr, specify the norm scale
    the same value as in script '03_3_get_simple_data_pkg_pos_audio_norm'
    function 'normalize_data_keep_dim(norm_max = norm_max)'
    '''
    
    rescale_arr = (norm_arr * (max_value - min_value) / norm_max) + min_value
    # rescale_arr = (norm_arr * (max_value - min_value)) + min_value
    
    return rescale_arr


def get_moving_average_2d(arr, n=3):
    '''
    from https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    get moving average along the time axis (axis 1) plus padding
    input: 2d arr [time, feature]
    output: 2d mean_arr [time, feature]
    '''

    arr = arr[np.newaxis, ...]

    ret = np.cumsum(arr, dtype=float, axis=1)
    ret[:, n:, :] = ret[:, n:, :] - ret[:, :-n, :]
    mean = ret[:, n - 1:, :] / n
    npad = ((0,0), (int(np.floor((n-1)/2)), int(np.ceil((n-1)/2))), (0,0))
    pad_mean = np.pad(mean, pad_width=npad, mode='edge')

    pad_mean = np.squeeze(pad_mean, axis = 0)
    
    return pad_mean


def get_recover_motion_pkg(pred_pkg):
    '''
    shift the pred/ ground truth motion position back to the mean position
    recover the normalization value
    '''

    recover_motion_pkg = {}

    motion_key_list = [
        'pred_pos_rhand', 'pred_pos_lhand', 'pred_pos_head',
        'truth_pos_rhand', 'truth_pos_lhand', 'truth_pos_head']
    
    min_max_key_list = [
        'min_max_rhand', 'min_max_lhand', 'min_max_head',
        'min_max_rhand', 'min_max_lhand', 'min_max_head'
    ]

    mean_key_list = [
        'mean_pos_rhand', 'mean_pos_lhand', 'mean_pos_head',
        'mean_pos_rhand', 'mean_pos_lhand', 'mean_pos_head'
    ]

    # recover the normalization scale
    for inx in range(len(motion_key_list)):
        key = motion_key_list[inx]
        motion_arr = pred_pkg[motion_key_list[inx]]
        min_value = pred_pkg[min_max_key_list[inx]][0]
        max_value = pred_pkg[min_max_key_list[inx]][1]
        mean_value = pred_pkg[mean_key_list[inx]]

        recover_motion_arr = transform_to_original_scale(
            norm_arr = motion_arr, 
            max_value = max_value, 
            min_value = min_value, 
            norm_max = motion_normalize_range)
        
        # shift to original mean position & smoothing
        recover_motion_arr = recover_motion_arr + mean_value
        recover_motion_arr = get_moving_average_2d(recover_motion_arr, n=11)

        recover_motion_pkg[key] = recover_motion_arr
    
    # get whole body mean pos and audio start time
    recover_motion_pkg['motion_pos_mean_all'] = pred_pkg['motion_pos_mean_all']
    recover_motion_pkg['audio_start_time'] = pred_pkg['audio_start_time']
    

    # check result
    print('=====')
    for key in recover_motion_pkg:
        print(key, recover_motion_pkg[key].shape)
        print(key, 'max: {:.2f}, min: {:.2f}, mean: {:.2f}'.format(
                np.nanmax(recover_motion_pkg[key]), 
                np.nanmin(recover_motion_pkg[key]),
                np.nanmean(recover_motion_pkg[key])))

    
    return recover_motion_pkg


def get_reconstruct_joint_pos(pred_motion_pkg, data_type):
    '''
    input: data_type = 'pred' or 'truth'
    motion_pkg from fn 'get_recover_motion_pkg'
    output: joint pos dict {joint_name: array[time, 3]}
    '''

    prefix = data_type + '_'
    
    # get prediction result
    con_head = np.array(pred_motion_pkg[prefix + 'pos_head'][:, 0:3])
    con_neck = np.array(pred_motion_pkg[prefix + 'pos_head'][:, 3:6])
    con_root = np.array(pred_motion_pkg[prefix + 'pos_head'][:, 6:9])

    con_btop = np.array(pred_motion_pkg[prefix + 'pos_rhand'][:, 0:3])
    con_relbow = np.array(pred_motion_pkg[prefix + 'pos_rhand'][:, 3:6])
    con_rwrist = np.array(pred_motion_pkg[prefix + 'pos_rhand'][:, 6:9])
    con_rfinger = np.array(pred_motion_pkg[prefix + 'pos_rhand'][:, 9:12])

    con_vtop = np.array(pred_motion_pkg[prefix + 'pos_lhand'][:, 0:3])
    con_lelbow = np.array(pred_motion_pkg[prefix + 'pos_lhand'][:, 3:6])
    con_lwrist = np.array(pred_motion_pkg[prefix + 'pos_lhand'][:, 6:9])
    con_lfinger = np.array(pred_motion_pkg[prefix + 'pos_lhand'][:, 9:12])

    print(con_lfinger.shape)

    # construct joints not in predictions
    # get the ref mean position
    joint_mean_dict = {}

    joint_mean_arr = pred_motion_pkg['motion_pos_mean_all']
    for inx in range(len(joint_list)):
        key = joint_list[inx]
        joint_mean_dict[key] = np.array(joint_mean_arr[inx*3: (inx+1)*3])
    
    # get the mean diff between joints
    diff_rshoulder_head = joint_mean_dict['Rshoulder'] - joint_mean_dict['Head']
    diff_lshoulder_head = joint_mean_dict['Lshoulder'] - joint_mean_dict['Head']
    diff_vbom_head = joint_mean_dict['Vbom'] - joint_mean_dict['Head']
    diff_bbom_rfinger = joint_mean_dict['Bbom'] - joint_mean_dict['Rfinger']

    # reconstruct joints
    con_rshoulder = con_head + diff_rshoulder_head
    con_lshoulder = con_head + diff_lshoulder_head
    con_vbom = con_head + diff_vbom_head
    con_bbom = con_rfinger + diff_bbom_rfinger

    # write result to dict
    recon_dict = {
        'Head': con_head,
        'Neck': con_neck,
        'Root': con_root,
        'Rshoulder': con_rshoulder,
        'Relbow': con_relbow,
        'Rwrist': con_rwrist,
        'Rfinger': con_rfinger,
        'Lshoulder': con_lshoulder,
        'Lelbow': con_lelbow,
        'Lwrist': con_lwrist,
        'Lfinger': con_lfinger,
        'Vtop': con_vtop,
        'Vbom': con_vbom,
        'Btop': con_btop,
        'Bbom': con_bbom
    }

    # for key in recon_dict.keys():
    #     print(key, recon_dict[key].shape)

    return recon_dict



# input joint_arr: structure array['joint_name'] = (3,)
def plot_skeleton_single_frame(
        joint_data_dict,
        joint_list, 
        joint_color_list, 
        joint_line_list,
        plt_time = 0, 
        legend = False):
    
    # take single frame

    joint_arr = {}
    for name in recon_joint_list:
        joint_arr[name] = joint_data_dict[name][plt_time, :]


    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")

    # Setting the axes properties
    ax.set(xlim3d=(750, -750), xlabel='X')
    ax.set(ylim3d=(750, -750), ylabel='Y')
    ax.set(zlim3d=(-50, 1500), zlabel='Z')    


    # plot joints
    for joint_inx in range(len(joint_list)):
        joint_name = joint_list[joint_inx]
        color = joint_color_list[joint_inx]
        joint_mean = joint_arr[joint_name]

        ax.scatter(joint_mean[0],
                   joint_mean[1], 
                   joint_mean[2],
                   c=color, marker='o', label=joint_name)

    # plot lines
    for line_inx in range(len(joint_line_list)):
        joint_name_1 = joint_list[joint_line_list[line_inx][0]]
        joint_name_2 = joint_list[joint_line_list[line_inx][1]]
        # print(joint_name_1, joint_name_2)
        joint_pos_1 = joint_arr[joint_name_1]
        joint_pos_2 = joint_arr[joint_name_2]

        ax.plot([joint_pos_1[0], joint_pos_2[0]],
                [joint_pos_1[1], joint_pos_2[1]],
                [joint_pos_1[2], joint_pos_2[2]], 
                c='black')


    # adding figure labels
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

    if legend == True:
        ax.legend(bbox_to_anchor=(1.1, 1.1), prop={'size':6}, ncol = 2)

    ax.set_title('Joint position')
    fig.tight_layout()
    ax.view_init(20, 100, 0)

    plt.show()




def animate_body_func(timeframe):
    '''
    input: joint_arr['joint_name'] = (time, xyz)
    from https://towardsdatascience.com/how-to-animate-plots-in-python-2512327c8263
    https://segmentfault.com/a/1190000020133415

    '''

    legend = False 
    fps=40
    
    # define plot data
    joint_list = [
        'Head','Neck','Root', 
        'Rshoulder', 'Relbow','Rwrist','Rfinger',
        'Lshoulder','Lelbow', 'Lwrist', 'Lfinger',
        'Vtop','Vbom','Btop', 'Bbom']

    joint_color_list = [
        'r', 'r', 'r',
        'b', 'b', 'b', 'b',
        'g', 'g', 'g', 'g',
        'black', 'black', 'black', 'black']

    joint_line_list = [
        [0, 1], [1, 2], [1, 3], [1, 7], 
        [3, 4], [4, 5], [5, 6],
        [7, 8], [8, 9], [9, 10],
        [11, 12], [13, 14]]

    
    # plot
    ax.clear()  # Clears the figure to update the line, point, title, and axes

    # updating joints
    for joint_inx in range(len(joint_list)):
        joint_name = joint_list[joint_inx]
        color = joint_color_list[joint_inx]
        joint_pos = plot_data_dict[joint_name] # (3,) xyz

        ax.scatter(
            joint_pos[timeframe, 0],
            joint_pos[timeframe, 1], 
            joint_pos[timeframe, 2],
            c=color, marker='o', label=joint_name)
    

    # updating lines
    for line_inx in range(len(joint_line_list)):
        joint_name_1 = joint_list[joint_line_list[line_inx][0]]
        joint_name_2 = joint_list[joint_line_list[line_inx][1]]
        joint_pos_1 = plot_data_dict[joint_name_1]
        joint_pos_2 = plot_data_dict[joint_name_2]

        ax.plot(
            [joint_pos_1[timeframe, 0], joint_pos_2[timeframe, 0]],
            [joint_pos_1[timeframe, 1], joint_pos_2[timeframe, 1]],
            [joint_pos_1[timeframe, 2], joint_pos_2[timeframe, 2]], 
            c='black')
    
    
    # # plot piano
    # piano_pos = [[600, 75, 0], [600, -75, 0], [-600, -75, 0], [-600, 75, 0]]
    # piano_line = [[0, 1], [1, 2], [2, 3], [3, 0]]

    # for piano_inx in range(len(piano_pos)):
    #     ax.scatter(piano_pos[piano_inx][0],
    #                piano_pos[piano_inx][1],
    #                piano_pos[piano_inx][2], 
    #                c = 'black')

    #     line_pos1 = piano_pos[piano_line[piano_inx][0]]
    #     line_pos2 = piano_pos[piano_line[piano_inx][1]]

    #     ax.plot([line_pos1[0], line_pos2[0]],
    #             [line_pos1[1], line_pos2[1]],
    #             [line_pos1[2], line_pos2[2]],
    #             c='black')
        
    # Setting Axes Limits
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_xlim(600, -600)
    ax.set_ylim(600, -600)
    ax.set_zlim(-400, 800)
    ax.view_init(20, 100, 0)

    # # Adding Figure Labels
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    
    if legend == True:
        ax.legend(bbox_to_anchor=(1.1, 1.1), prop={'size':6}, ncol = 2)
    
    # ax.legend(bbox_to_anchor=(1.35, 1)) # (1, 0.5)
    time = np.round(timeframe/fps, decimals=0)
    # ax.set_title('Joint position \nTime =' + '{:.0f}'.format(time) +  ' sec')
    # ax.set_title('Joint position')


def plot_animation(plot_data_dict):
    
    global ax

    print('Processing animation ...')
    fig = plt.figure(figsize=(6,6))
    ax = plt.axes(projection='3d')
    num_time_frame = plot_data_dict['Head'].shape[0]

    body_ani = animation.FuncAnimation(
        fig, animate_body_func, 
        interval= 1000/40,
        frames=num_time_frame)

    # save animation
    # FFwriter = animation.PillowWriter(fps=40)
    FFwriter = animation.FFMpegWriter(fps=40)

    ani_name = song_name + '_' + perf_name + '_' + trial_name + '_animation_temp.mp4'
    ani_file =  os.path.join(script_folder, 'temp_folder', ani_name)
    body_ani.save(ani_file, writer=FFwriter)
    # print('Save animation to:', ani_file)

    # add audio    
    data_folder = os.path.join(parent_folder, 'sample_data', dataset_name)
    audio_file = os.path.join(data_folder, song_name, perf_name, trial_name, song_name + '_' + perf_name + '_' + trial_name + '_audio.wav')
    audio_clip_file = os.path.join(script_folder, 'temp_folder', song_name + '_' + perf_name + '_' + trial_name + '_audio_clip.wav')
    audio_start_time = recover_pos_pkg['audio_start_time']

    final_animation_file = os.path.join(script_folder, song_name + '_' + perf_name + '_' + trial_name + '_animation_w_audio.mp4')

    # make audio clip
    audio, sr = librosa.load(audio_file, sr=22050, offset=audio_start_time, duration=16)
    sf.write(audio_clip_file, audio, 22050)

    # Open the video and audio
    video_clip = VideoFileClip(ani_file)
    audio_clip = AudioFileClip(audio_clip_file)

    # Concatenate the video clip with the audio clip
    final_clip = video_clip.set_audio(audio_clip)
    # final_clip = moviepy.audio.fx.all.volumex(final_clip, 8) # enhence the volume

    # Export the final video with audio
    final_clip.write_videofile(final_animation_file)





    



if __name__ == '__main__':

    # Define test audio 
    # This scipt is for violin animation only (not for piano)
    # Please select an audio example from 'yv' and 'ev' subsets ('yp' is a piano subset)
    # test_audio_name = 'de2_yv10_t1_audio.wav' # define audio_name
    parser = ArgumentParser()
    parser.add_argument(
        "audiofilename", 
        nargs="?", 
        help="audio file name, ex:de2_yv10_t1_audio.wav", 
        default="de2_yv10_t1_audio.wav")

    args = parser.parse_args()
    test_audio_name = args.audiofilename


    # Define test dataset
    # Select from the folders 'MOSA_dataset' or 'sample_data'
    # 'MOSA-dataset' contains all data, while 'sample_data' contains only few audio examples
    test_dataset_name = 'sample_data' 

    # Get file directories
    [song_name, perf_name, trial_name, name4] = test_audio_name.split("_")
    dataset_name = perf_name[:2]
    script_path = os.path.abspath(__file__)
    script_folder = os.path.abspath(os.path.join(script_path, os.pardir))
    parent_folder = os.path.abspath(os.path.join(script_folder, os.pardir))
    data_folder = os.path.join(parent_folder, test_dataset_name, dataset_name)
    audio_file = os.path.join(data_folder, song_name, perf_name, trial_name, test_audio_name)

    # Define eval model directory
    model_name = os.path.join(script_folder, 'pre_trained_model_motion_generation_violin')


    # Define save animation directory
    save_ani_dir = script_folder
    mpeg_file = os.path.join(parent_folder, 'utilities', 'ffmpeg-6.0-essentials_build', 'bin', 'ffmpeg.exe')
    plt.rcParams['animation.ffmpeg_path'] = mpeg_file
    matplotlib.rcParams['animation.embed_limit'] = 2**128


    # load parameters
    load_parameters()
    load_violin_joint_marker_list()
    load_joint_list()
    load_recon_joint_list()

    # get test data
    test_data_pkg = get_song_data_label_snippet_list(
        data_folder,
        song_name,
        perf_name,
        trial_name,
        audio_range = (0, 10),
        if_audio_norm = True
        )
    
    test_data = get_test_label_from_pkg(test_data_pkg)
    test_data = get_test_data_from_pkg(test_data)

    # get prediction
    pred_result = get_prediction(model_name, test_data)
    recover_pos_pkg = get_recover_motion_pkg(pred_result)
    pred_joint_dict = get_reconstruct_joint_pos(recover_pos_pkg, data_type = 'pred')
    truth_joint_dict = get_reconstruct_joint_pos(recover_pos_pkg, data_type = 'truth')
    
    # # plot still image of prediction
    # plot_skeleton_single_frame(
    #     pred_joint_dict,
    #     joint_list = recon_joint_list, 
    #     joint_color_list = recon_joint_color_list, 
    #     joint_line_list = recon_joint_line_list,
    #     plt_time = 0,  
    #     legend = False)

    # Plot body motion Animation
    plot_data_dict = pred_joint_dict
    plot_animation(plot_data_dict)




    
