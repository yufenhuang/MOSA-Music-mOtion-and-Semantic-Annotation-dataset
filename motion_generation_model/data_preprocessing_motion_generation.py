import joblib, csv, os
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
import sklearn.preprocessing
import librosa

np.set_printoptions(suppress=True, precision=3)


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
                    if dyn_onset <= note_annot_dict[note_inx]['score_position'] and note_annot_dict[note_inx]['score_position'] < dyn_offset:
                        note_annot_dict[note_inx]['dyn'] = dyn_name

        # arti annot
        for arti_name in arti_type_list:
            if express_item['expression'] == arti_name:
                arti_onset = express_item['onset']
                arti_offset = express_item['offset']

                for note_inx in range(len(note_annot_dict)):
                    if arti_onset <= note_annot_dict[note_inx]['score_position'] and note_annot_dict[note_inx]['score_position'] < arti_offset:
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
    print('start time: {:.3f}'.format(start_time))
    print('end time: {:.3f}'.format(end_time))

    time_annot_list = []
    for time in time_list:
        frame = {
            'time': time, 'beat': 0, 'downbeat': 0, 'phrase': 0, 'dyn': 'none', 'arti': 'none', 'pitch': 'none', 'midi': np.nan
            }

        # get frame dyn & arti label
        for note in note_annot_dict:
            if note['onset'] <= time and time < note['offset']:
                frame['dyn'] = note['dyn']
                frame['arti'] = note['articu']
            
        # get frame beat, downbeat, phrase label
        for beat_time in perf_beat_time_list:
            if beat_time - beat_aug_time <= time and time <= beat_time + beat_aug_time:
                frame['beat'] = 1
        
        for downbeat_time in perf_downbeat_time_list:
            if downbeat_time - downbeat_aug_time <= time and time <= downbeat_time + downbeat_aug_time:
                frame['downbeat'] = 1
        
        for phrase_time in perf_phrase_time_list:
            if phrase_time - phrase_aug_time <= time and time <= phrase_time + phrase_aug_time:
                frame['phrase'] = 1
        
        # get frame pitch
        for note in note_annot_dict:
            if note['onset'] <= time and time < note['offset']:
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

def load_violin_joint_marker_list():

    global marker_name_list, new_joint_list, joint_name_list, norm_ref_list

    # original 34 markers list
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
   
    
    # output 21 joint data
    joint_name_list =  [
        'Head','Neck','Root', 
        'Rshoulder', 'Relbow','Rwrist','Rfinger',
        'Lshoulder','Lelbow', 'Lwrist', 'Lfinger',
        'Rknee','Rank','Rtoe', 
        'Lknee','Lank','Ltoe',
        'Vtop','Vbom','Btop', 'Bbom'
        ]

    # the reference joints used for normalization
    norm_ref_list = [['Rtoe', 72, 75], ['Ltoe', 84, 87]]




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

    return audio_clip



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


def remove_data_nan(data_arr, axis=0):
    data_arr = pd.DataFrame(data_arr)
    data_arr = data_arr.interpolate(axis= axis, limit_direction= 'both')
    data_arr = np.array(data_arr)

    return data_arr


def check_nan(data_arr):
    nan_arr = np.isnan(data_arr)

    if True in nan_arr:
        nan_num = np.count_nonzero(nan_arr == True)
        # arr_sample = data_arr[0, :]
        # arr_sample2 = data_arr[1, :]
        print('Warning: NaN in data array !!!')
        print('Number of NaN:', nan_num)
        # for inx in range(arr_sample.shape[0]):
        #     print(inx, arr_sample[inx], arr_sample2[inx])

    elif True not in nan_arr:
        print('no Nan in data')



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


def rescale_audio_data(data_arr, rescale_range=(0, 10)):
    '''
    input: data array
    rescale_range: (lower_bound, upper_bound)
    '''
    scaler = sklearn.preprocessing.MinMaxScaler(rescale_range)
    new_data_arr = scaler.fit_transform(data_arr)
    print('rescale data to range:', rescale_range)

    return new_data_arr




def get_snippet(
        audio_data, 
        position_data,
        velocity_data,
        flux_data, 
        rms_data,
        label_list,
        id_name,
        snippet_time_window = 16,
        sampling_rate = 40,
        snippet_hop_size = 2,
        if_snippet_hop = True
        ):

    '''
    input: audio_data [time, feature], pos_data [time, feature], velocity_data [time, feature]
    label_list: list {'beat', 'downbeat', 'phrase', 'dyn', 'arti'} 
    snippet_time_window: in second
    snippet_hop_size: in second
    sampling_rate: frame per second
    
    return: snippet list {'audio_data', 'motion_data', 'beat_label', 'downbeat_label', 'phrase_label', 'dyn_label', 'arti_label'} 
    '''

    if if_snippet_hop == True:
        snippet_hop_size = snippet_hop_size
    elif if_snippet_hop == False:
        snippet_hop_size = snippet_time_window
    

    snippet_list = []

    # get stride list
    n_frame_window = int(snippet_time_window * sampling_rate)
    n_frame_hop = int(snippet_hop_size * sampling_rate)
    start_frame = 0
    end_frame = np.min([len(label_list), 
                        audio_data.shape[0], 
                        position_data.shape[0], 
                        velocity_data.shape[0]]) - n_frame_window
    stride_list = np.arange(start_frame, end_frame, n_frame_hop)
    # print('stride step', len(stride_list), stride_list)

    # get snippet
    for seg_inx in range(len(stride_list)):
        frame_inx = stride_list[seg_inx]
        win_start = frame_inx
        win_end = frame_inx + n_frame_window
        audio_seg = audio_data[win_start:win_end, :]
        pos_seg = position_data[win_start:win_end, :]
        vel_seg = velocity_data[win_start:win_end, :]
        flux_seg = flux_data[win_start:win_end, :]
        rms_seg = rms_data[win_start:win_end, :]
        label_seg = label_list[win_start:win_end]
        seg_id = id_name + '_' + str(seg_inx)
        # print('start & end', win_start, win_end, win_end - win_start)

        # get annot
        beat_seg = np.array([item['beat'] for item in label_seg])
        downbeat_seg = np.array([item['downbeat'] for item in label_seg])
        phrase_seg = np.array([item['phrase'] for item in label_seg])
        dyn_seg = np.array([item['dyn'] for item in label_seg])
        arti_seg = np.array([item['arti'] for item in label_seg])
        midi_seg = np.array([item['midi'] for item in label_seg])
        midi_seg =  remove_data_nan(midi_seg, axis=0)

        snippet = {
            'audio_data': audio_seg,
            'motion_pos_data': pos_seg,
            'motion_vel_data': vel_seg,
            'beat_label': beat_seg,
            'downbeat_label':downbeat_seg,
            'phrase_label': phrase_seg,
            'dyn_label': dyn_seg,
            'arti_label': arti_seg,
            'midi_label': midi_seg,
            'flux': flux_seg,
            'rms': rms_seg,
            'id': seg_id
        }

        snippet_list.append(snippet)

    return snippet_list



def get_song_data_label_snippet_list(
        folder_dir,
        song_name,
        perf_name,
        trial_name,
        snippet_hop_size = 2,
        audio_range = (0, 10),
        velocity_range = (-50, 50),
        if_snippet_hop = True,
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
        if_aug_time = if_aug_time
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

    song_audio_data = get_audio_data(
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
        
    # get snippet
    song_snippet_list = get_snippet(
        audio_data = song_audio_data, 
        position_data = song_joint_data,
        velocity_data = song_velocity_data,
        flux_data = song_flux_data,
        rms_data = song_rms_data,
        label_list = song_annot_list,
        id_name = song_name + '_' + perf_name + '_' + trial_name,
        snippet_time_window = snippet_time_window,
        sampling_rate = data_sampling_rate,
        snippet_hop_size = snippet_hop_size,
        if_snippet_hop = if_snippet_hop)
    
    return song_snippet_list


    

# def get_data_pkg_from_folder(
#         folder_dir, 
#         train_test_split_ratio = 10
#         ):

#     train_pkg = []
#     test_pkg = []
#     song_count = 0

#     # search for subfolders
#     for song_name in os.listdir(folder_dir):
#         song_folder = os.path.join(folder_dir, song_name)

#         for perf_name in os.listdir(song_folder):
#             perf_folder = os.path.join(song_folder, perf_name)

#             for trial_name in os.listdir(perf_folder):
#                 trial_folder = os.path.join(perf_folder, trial_name)
#                 song_file_name = song_name + '_' + perf_name + '_' + trial_name
#                 song_count += 1
#                 print('-----')
#                 print('Data name: ', song_file_name)

#                 # split to training or test set
#                 # get song snippet for test package
#                 if song_count % train_test_split_ratio == 0:
#                     pkg_name = 'test'

#                     song_snippet = get_song_data_label_snippet_list(
#                         folder_dir,
#                         song_name,
#                         perf_name,
#                         trial_name,
#                         audio_range = audio_normalize_range,
#                         if_snippet_hop = False,
#                         if_aug_time = False)

#                     for item in song_snippet:
#                         test_pkg.append(item)
                

#                 # save song snippet to train package
#                 else:
#                     pkg_name = 'train'

#                     song_snippet = get_song_data_label_snippet_list(
#                         folder_dir,
#                         song_name,
#                         perf_name,
#                         trial_name,
#                         audio_range = audio_normalize_range,
#                         if_snippet_hop = True,
#                         if_aug_time = True)

#                     for item in song_snippet:
#                         train_pkg.append(item)
                 
#                 print('Assign to: ', pkg_name, 'set')

#     # print total data of packages
#     print('==========')
#     print('----- Preprocessing data package done -----')
#     print('Data shape for each snippet:')
#     for key in train_pkg[0].keys():
#         if key != 'id':
#             print(key, train_pkg[0][key].shape)
    
#     print('Training data num:', len(train_pkg))
#     print('Testing data num:', len(test_pkg))
                
#     return train_pkg, test_pkg







def get_data_pkg_from_folder(folder_dir):

    train_pkg = []
    test_pkg = []
    song_count = 0


    # define dataset and subset
    if instrument_name == 'violin':
        subset_list = ['yv', 'ev']
    elif instrument_name == 'piano':
        subset_list = ['yp']
    else:
        print('Invalid insrument_name')
        print('please select from \"violin\" or \"piano\"')
    

    if dataset_name == 'sample_data':
        train_test_split_ratio = 2
        
    elif dataset_name == 'MOSA_dataset':
        train_test_split_ratio = 10

    else:
        print('Invalid dataset name')
        print('please select from \"sample_data\" or \"MOSA_dataset\"')


    # search for subfolders
    for subset_name in subset_list:
        if os.path.exists(os.path.join(folder_dir, subset_name)) == True:
            subset_dir = os.path.join(folder_dir, subset_name)

            for song_name in os.listdir(subset_dir):
                try:
                    song_folder = os.path.join(subset_dir, song_name)
                except:
                    continue

                for perf_name in os.listdir(song_folder):
                    try:
                        perf_folder = os.path.join(song_folder, perf_name)
                    except:
                        continue

                    for trial_name in os.listdir(perf_folder):

                        song_id_name = subset_name + '_' + song_name + '_' + perf_name + '_' + trial_name
                        print('-----')
                        print('Data name: ', song_id_name)


                        # skip problem files
                        # check_file = check_problem_files(song_id_name, problem_files)
                        # if check_file == True:
                        if song_id_name in problem_files:
                            print('SKIP PROBLEMATIC FILES')
                            break

                        elif song_name in problem_songs:
                            print('SKIP PROBLEMATIC songs')
                            break
                            
                        else:
                            trial_folder = os.path.join(perf_folder, trial_name)
                            song_file_name = song_name + '_' + perf_name + '_' + trial_name
                            song_count += 1
                            

                            # split to training or test set
                            # get song snippet for test package
                            if song_count % train_test_split_ratio == 0:
                                pkg_name = 'test'

                                song_snippet = get_song_data_label_snippet_list(
                                    subset_dir,
                                    song_name,
                                    perf_name,
                                    trial_name,
                                    audio_range = audio_normalize_range,
                                    if_snippet_hop = False,
                                    if_aug_time = False)

                                for item in song_snippet:
                                    test_pkg.append(item)
                            

                            # save song snippet to train package
                            else:
                                pkg_name = 'train'

                                song_snippet = get_song_data_label_snippet_list(
                                    subset_dir,
                                    song_name,
                                    perf_name,
                                    trial_name,
                                    audio_range = audio_normalize_range,
                                    if_snippet_hop = True,
                                    if_aug_time = True)

                                for item in song_snippet:
                                    train_pkg.append(item)
                            
                            print('Assign to: ', pkg_name, 'set')


    # print total data of packages
    print('==========')
    print('----- Preprocessing data package done -----')
    print('Data shape for each snippet:')
    for key in train_pkg[0].keys():
        if key != 'id':
            print(key, train_pkg[0][key].shape)
    
    print('Training data num:', len(train_pkg))
    print('Testing data num:', len(test_pkg))
                
    return train_pkg, test_pkg











if __name__ == '__main__':



    # ===== define processed data =====
    # select from 'violin'  or 'piano' 
    # 'violin' will process 'yv' and 'ev' subsets, while 'piano' will process 'yp' subset
    instrument_name = 'violin' 

    # Select from the folders 'MOSA_dataset' or 'sample_data'
    # 'MOSA-dataset' contains all data, while 'sample_data' contains only few audio examples
    dataset_name = 'sample_data' # select from 'sample_data' or 'MOSA_dataset'


    # ===== define data folder =====
    script_path = os.path.abspath(__file__)
    script_folder = os.path.abspath(os.path.join(script_path, os.pardir))
    parent_folder = os.path.abspath(os.path.join(script_folder, os.pardir))
    data_folder = os.path.join(parent_folder, dataset_name)
    print('Input data from folder:', data_folder)


    # ===== define parameters for data processing =====
    data_sampling_rate = 40
    beat_label_augment_time = 0.05 # beat label extend +- n seconds
    downbeat_label_augment_time = 0.05 # downbeat label extend +- n seconds
    phrase_label_augment_time = 0.5 # phrase label extend +- n seconds
    body_marker_num = 30 # 30 for motion generation, 22 for time/expression semantics
    instrument_marker_num = 4 # 4 for motion generation, 0 for time/expression semantics
    snippet_time_window = 16
    snippet_hop_size = 2
    audio_normalize_range = (0, 30)
    train_test_data_ratio = 2 # 10
    problem_files = []
    problem_songs = []

    
    # ===== process data =====
    train_data_pkg, test_data_pkg = get_data_pkg_from_folder(data_folder)

    
    # ===== save processed data as pickle =====
    save_pkl_folder = script_folder
    train_pkl_file = os.path.join(save_pkl_folder, instrument_name + "_training_data_pkg.pkl")
    test_pkl_file = os.path.join(save_pkl_folder, instrument_name + "_testing_data_pkg.pkl")
    
    joblib.dump(train_data_pkg, train_pkl_file)
    joblib.dump(test_data_pkg, test_pkl_file)
    
    print('----- Save pre-processed data package done -----')
    print('Data pkg keys:', train_data_pkg[0].keys())
    print('Save training data package to:', train_pkl_file)
    print('Save testing data package to:', test_pkl_file)



    




    
