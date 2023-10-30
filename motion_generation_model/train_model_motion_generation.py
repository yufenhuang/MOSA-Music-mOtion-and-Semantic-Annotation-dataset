# supress numpy future warning
import warnings
warnings.filterwarnings('ignore')
import librosa, os, sys, pickle, copy, mir_eval, joblib, datetime
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_nlp
from scipy import signal
from sklearn import preprocessing
from tensorflow import keras
from keras import layers

# show version info
print ("[info] Current Time:     " + datetime.datetime.now().strftime('%Y/%m/%d  %H:%M:%S'))
print ("[info] Python Version:   " + sys.version.split('\n')[0].split(' ')[0])
print ("[info] Working Dir:      " + os.getcwd())
print ("[info] Tensorflow:       " + tf.__version__)
print ("[info] Keras:            " + keras.__version__)

# limit GPU memory usage
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

np.set_printoptions(suppress=True)



def split_train_eval_sub_pkg(data_pkg, train_eval_split_ratio):
    '''
    split data_pkg to train_pkg & eval_pkg
    input: data_pkg list {key: data_arr}
    output: train_pkg list {key: data_arr},
            eval_pkg list {key: data_arr}
    '''

    train_pkg = [] 
    eval_pkg = []

    for inx in range(len(data_pkg)):

        if inx % train_eval_split_ratio == 0:
            eval_pkg.append(data_pkg[inx])
            # print(inx, 'assign to eval pkg')
        
        elif inx % train_eval_split_ratio != 0:
            train_pkg.append(data_pkg[inx])
            # print(inx, 'assign to train pkg')
    
    print('train pkg:', len(train_pkg))
    print('eval pkg:', len(eval_pkg))

    del data_pkg
    return train_pkg, eval_pkg


def get_data_and_label_from_pkg(data_pkg):
    '''
    reorganize data and label as array for input data package
    input: data_pkg: list {keys: array}
    output: data_pkg dict {keys, array}
    '''

    # shuffle data & label
    data_pkg = random.sample(data_pkg, len(data_pkg))

    # get data & label array
    load_data_pkg = {}

    for key in data_pkg[0].keys(): 

        try:
            data_arr = np.array([item[key] for item in data_pkg], dtype=np.float32)
        
        except: 
            data_arr = np.array([item[key] for item in data_pkg])
        
        # clean up data
        if key == 'audio_data' or key == 'motion_pos_data' or key == 'motion_vel_data':
            new_data_arr = remove_nan_3d_array(data_arr, axis=1)
            new_data_arr = replace_nan_by_mean_3d_array(data_arr)
            # print(key, 'processing audio, motion data')
            
        # convert string label to class label
        elif key == 'dyn_label':

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
        # print('data type:', load_data_pkg[key].dtype)

        if key == 'audio_data' or key == 'motion_pos_data' or key == 'motion_vel_data':
            print(key, 'max: {:.2f}, min: {:.2f}, mean: {:.2f}'.format(
                np.nanmax(load_data_pkg[key]), 
                np.nanmin(load_data_pkg[key]),
                np.nanmean(load_data_pkg[key])))
        
    del data_pkg
    return load_data_pkg
    


def get_motion_generation_pkg(data_pkg, norm_range = 1):

    new_data_pkg = {}

    # get audio data
    new_data_pkg['audio_data'] = data_pkg['audio_data']

    # combine multiple labels
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

    # get body segment data
    pos_head_data, pos_rhand_data, pos_lhand_data = get_body_seg_data_from_joint_arr(data_pkg['motion_pos_data'])
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
        new_data_pkg[motion_key], new_data_pkg[mean_key_list[inx]] = shift_pos_to_joint_mean(new_data_pkg[motion_key])
        new_data_pkg[motion_key] = clean_outlier_keep_dim(
            new_data_pkg[motion_key], 
            tolerance = 1.5, keep_dim = 2)
        
        # normalize the range of motion data
        new_data_pkg[motion_key], data_min, data_max = normalize_data_all_dim(
            new_data_pkg[motion_key], norm_max = norm_range)
        new_data_pkg[min_max_key_list[inx]] = np.array([data_min, data_max])
    

    # check data
    print('---')  
    print('Normalized data:')
    for key in new_data_pkg.keys():
        print(key, new_data_pkg[key].shape)
        # print('data type:', load_data_pkg[key].dtype)

        if key == 'motion_pos_head' or key == 'motion_pos_rhand' or key == 'motion_pos_lhand':
            print(key, 'max: {:.2f}, min: {:.2f}, mean: {:.2f}'.format(
                np.nanmax(new_data_pkg[key]), 
                np.nanmin(new_data_pkg[key]),
                np.nanmean(new_data_pkg[key])))

    del data_pkg
    return new_data_pkg


def convert_str_to_one_hot_label(
        str_label_arr,
        label,
        one_hot):
    '''
    convert str label array to one hot array
    input: str_label_arr [batch, time]
    label: list of string label, 
    e.g., ['ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff']
    ['legato', 'staccato']
    one_hot: correspond value of label
    e.g., [0,0,1,2,3,4,5,5], [0,1]

    output: one_hot_label_arr [batch, time , n_class]
    '''

    int_label_arr = np.zeros((str_label_arr.shape[0], str_label_arr.shape[1]))

    for name, value in zip(label, one_hot):
        # print(name, value)
        int_label_arr[str_label_arr == name] = value
    # print(int_label_arr[:10, :5])

    one_hot_arr = (np.arange(np.array(one_hot).max() + 1) == int_label_arr[...,None]).astype(float)
    # print(one_hot_arr[:10, :5, :])
    print('convert label array to one hot array')
    print('label array:', int_label_arr.shape)
    print('one hot array:', one_hot_arr.shape)

    return one_hot_arr


def convert_str_to_class_label(
        str_label_arr,
        label,
        class_name):

    class_label_arr = np.zeros(str_label_arr.shape).astype(float)

    for inx in range(len(label)):
        item_inx = np.where(str_label_arr == label[inx])
        class_label_arr[item_inx] = class_name[inx]
    
    return class_label_arr



def remove_nan_3d_array(data_arr, axis=1):
    '''
    replace nan for 3d array
    input: data_arr [batch, time, feature]
    '''

    new_data_arr = np.zeros_like(data_arr)

    for batch_inx in range(data_arr.shape[0]):
        data_slice = data_arr[batch_inx, :, :]
        data_slice = pd.DataFrame(data_slice)
        data_slice = data_slice.interpolate(axis= int(axis-1), limit_direction= 'both')
        data_slice = np.array(data_slice)
        new_data_arr[batch_inx, :, :] = data_slice

    return new_data_arr


def replace_nan_by_mean_3d_array(data_arr):
    '''
    replace 3d data array by the data mean of feature (axis 2)
    input: data_arr [batch, time, feature]
    output: new_data_arr [batch, time, feature]
    '''

    mean_arr = np.nanmean(data_arr, axis=(0, 1))
    nan_arr = np.isnan(data_arr)
    nan_inx = np.where(nan_arr == True)
    # print('replace by mean value:', mean_arr)
    # print('nan inx:', nan_inx)
    
    for inx_0, inx_1, inx_2 in zip(nan_inx[0], nan_inx[1], nan_inx[2]):
        marker_mean = mean_arr[inx_2]
        data_arr[inx_0, inx_1, inx_2] = marker_mean
        
    return data_arr


def get_body_seg_data_from_joint_arr(joint_arr):

    rhand = joint_arr[:, :, 12:21]
    lhand = joint_arr[:, :, 24:33]
    btop = joint_arr[:, :, 57:60]
    vtop = joint_arr[:, :, 51:54]

    pos_head = np.array(joint_arr[:, :, :9])
    pos_rhand = np.concatenate((btop, rhand), axis=-1)
    pos_lhand = np.concatenate((vtop, lhand), axis=-1)

    return pos_head, pos_rhand, pos_lhand


def shift_pos_to_joint_mean(motion_arr):
    '''
    shift each joint to its own mean position
    get the mean pos of each joint's xyz
    input: 3d motion_arr [batch, time, feature (joint*xyz)]
    output: shifted joint arr pos_shift [batch, time, feature (joint*xyz)]
    the mean position of each joint pos_mean [feature (joint*xyz)]
    '''

    pos_mean = np.mean(motion_arr, axis= (0, 1))
    pos_shift = np.array(motion_arr - pos_mean)

    return pos_shift, pos_mean


def clean_outlier_keep_dim(data_arr, tolerance = 1.5, keep_dim = 2):
    '''
    clean outliers for 3d input arr
    keep one dimension
    
    input: data_arr[clip, time, feature]
    
    output:clean_arr [clip, time, feature]
    
    for motion data: keep_dim = 2 (feature, xyz)
    tolerance: Q75/Q25  +- Quartile*tolerance
    '''

    all_dim = [0, 1, 2]
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

        f_arr = data_arr[:, :, f_inx]
        f_arr = np.where(f_arr > f_upper, f_upper, f_arr)
        f_arr = np.where(f_arr < f_lower, f_lower, f_arr)
        clean_arr[:, :, f_inx] = f_arr
        clean_arr = np.array(clean_arr)
    
    return clean_arr


def normalize_data_all_dim(data_arr, norm_max = 1):
    '''
    normalize 3D input arr to the range [0, norm_max]
    
    input: data_arr[batch, time, feature]
    
    output:norm_data [batch, time, feature]
    data_min
    data_max
    
    '''
                                                    
    data_min = np.min(data_arr)
    data_max = np.max(data_arr)
    
    norm_data = np.array((data_arr - data_min) * norm_max / (data_max - data_min))
    
    return norm_data, data_min, data_max


def load_data_pkg_fn(data_pkl_file, 
                     train_eval_split_ratio = 10, 
                     norm_range = 1):

    input_data_pkg = joblib.load(data_pkl_file)
    print('Load data keys:', input_data_pkg[0].keys())
    print('Load data len:', len(input_data_pkg))

    train_data_pkg, eval_data_pkg = split_train_eval_sub_pkg(
        input_data_pkg, 
        train_eval_split_ratio = train_eval_split_ratio
        )
    
    del input_data_pkg

    print('=====')
    print('get train data:')
    train_data = get_data_and_label_from_pkg(train_data_pkg)
    train_data = get_motion_generation_pkg(train_data, norm_range = norm_range)

    print('=====')
    print('get eval data:')
    eval_data = get_data_and_label_from_pkg(eval_data_pkg)
    eval_data = get_motion_generation_pkg(eval_data, norm_range = norm_range)

    del train_data_pkg
    del eval_data_pkg
    
    return train_data, eval_data



class DataGenerator_motion_generation(tf.keras.utils.Sequence):
    """
    Custom data generator class for Digits dataset
    input: input data label pkg
    output: [input_data], [input_labels]
    """

    def __init__(self, data, batch_size=32):
        self.audio = data['audio_data'] 
        self.label = data['comb_label']
        self.pos_rhand = data['motion_pos_rhand']
        self.pos_lhand = data['motion_pos_lhand']
        self.pos_head = data['motion_pos_head']
        self.batch_size = batch_size
        
    
    def __len__(self):
        return int(np.math.floor(self.audio.shape[0] / self.batch_size))
    
    def __getitem__(self, index):
        """
        Returns a batch of data
        """       
        # Generate indexes of the batch
        ind_start = index*self.batch_size
        inx_end = (index+1)*self.batch_size
        
        batch_audio = self.audio[ind_start:inx_end, :, :]
        batch_label = self.label[ind_start:inx_end, :, :]
        batch_pos_rhand = self.pos_rhand[ind_start:inx_end, :]
        batch_pos_lhand = self.pos_lhand[ind_start:inx_end, :]
        batch_pos_head = self.pos_head[ind_start:inx_end, :]
        
        # return index, ind_start, inx_end
        
        return [batch_audio, batch_label], [batch_pos_rhand, batch_pos_lhand, batch_pos_head]



class ConvEmbedding(layers.Layer):
    '''
    Define data embedding layer
    '''
    
    def __init__(self, num_hid, maxlen=640, kernal_size=11, stride=1):
        super().__init__()
        # self.conv1 = layers.Conv1D(
        #     num_hid, kernal_size, strides=stride, padding="same", activation="relu"
        # )
        # self.conv2 = layers.Conv1D(
        #     num_hid, kernal_size, strides=stride, padding="same", activation="relu"
        # )
        self.conv3 = layers.Conv1D(
            num_hid, kernal_size, strides=stride, padding="same", activation="relu"
        )
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)
        self.maxlen = maxlen

    def call(self, x):
        # x = self.conv1(x)
        # x = self.conv2(x)
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        output = self.conv3(x) + positions
        return output



def build_transformer_model():

    # input
    audio_inputs = layers.Input(
        shape=(time_frame_size, audio_feature_size), 
        name='audio_input') # (time, feature)
    feature_inputs = layers.Input(
        shape=(time_frame_size, label_feature_size), 
        name='feature_input')
        

    # embedding
    audio_embedding_layer = ConvEmbedding(num_hid=audio_embed_dim, maxlen=time_frame_size)
    feature_embedding_layer = ConvEmbedding(num_hid=label_embed_dim, maxlen=time_frame_size)

    audio_data_embedding = audio_embedding_layer(audio_inputs)
    feature_data_embedding = feature_embedding_layer(feature_inputs)

    total_embedding = tf.concat(
        [audio_data_embedding, feature_data_embedding], 
        axis=-1, name='concat_embed')

    # ===== main encoder =====
    for en_inx in range(num_layers_main_enc):

        encoder_layer = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=ff_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            activation="relu",
            name='main_encoder' + str(en_inx+1)
        )
        
        # for the first decoder layer
        if en_inx == 0:
            main_branch = encoder_layer(total_embedding)
        else:
            main_branch = encoder_layer(main_branch)

    encoder_output = main_branch


    # ===== branch encoder =====
    # ===== rhand_encoder =====
    for en_inx in range(num_layers_branch_enc):
        
        encoder_layer = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=ff_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            activation="relu",
            name='rhand_encoder' + str(en_inx+1)
        )
        
        # for the first decoder layer
        if en_inx == 0:
            rhand_branch = encoder_layer(main_branch)
        else:
            rhand_branch = encoder_layer(rhand_branch)

    # rhand_output
    # beat_outputs = layers.Dense(output_dim, activation="sigmoid")(beat_branch)
    rhand_outputs = layers.Dense(rhand_output_dim, activation="linear", name = 'rhand_dense')(rhand_branch)


    # ===== lhand_decoder =====
    for en_inx in range(num_layers_branch_enc):

        encoder_layer = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=ff_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            activation="relu",
            name='lhand_encoder' + str(en_inx+1)
        )
        
        # for the first decoder layer
        if en_inx == 0:
            lhand_branch = encoder_layer(main_branch)
        else:
            lhand_branch = encoder_layer(lhand_branch)

    # lhand_output
    # downbeat_outputs = layers.Dense(output_dim, activation="sigmoid")(downbeat_branch)
    lhand_outputs = layers.Dense(lhand_output_dim, activation="linear", name = 'lhand_dense')(lhand_branch)


    # ===== head_decoder =====
    for en_inx in range(num_layers_branch_enc):

        encoder_layer = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=ff_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            activation="relu",
            name='head_encoder' + str(en_inx+1)
        )
        
        # for the first decoder layer
        if en_inx == 0:
            head_branch = encoder_layer(main_branch)
        else:
            head_branch = encoder_layer(head_branch)

    # head_output
    # phrase_outputs = layers.Dense(output_dim, activation="sigmoid")(phrase_branch)
    head_outputs = layers.Dense(head_output_dim, activation="linear", name = 'head_dense')(head_branch) # "exponential"


    # build model
    model = keras.Model([audio_inputs, feature_inputs], 
                        [rhand_outputs, lhand_outputs, head_outputs], name="motion_generation_transformer")    
    model.summary()

    return model



# define loss

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



def training(model, train_subset, eval_subset):

    # learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)

    # compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule) 

    model.compile(
        optimizer=optimizer,
        loss= custom_mse_loss_fn, # mse_2d
        loss_weights={
            'rhand_dense': rhand_loss_weight, 
            'lhand_dense': rhand_loss_weight,
            'head_dense': head_loss_weight}, 
            metrics=[tf.keras.metrics.MeanSquaredError()]) 

    print('compile model done!')

    # get batch data from generator
    train_batch_data = DataGenerator_motion_generation(train_subset, batch_size=batch_size)
    eval_batch_data = DataGenerator_motion_generation(eval_subset, batch_size=batch_size)
    print('training data batch:', len(train_batch_data))
    print('eval data batch:', len(eval_batch_data))

    # save model for n epoches
    batch_per_epoch = len(train_batch_data)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath= os.path.join(save_model_dir, 'motion_generation_transformer', "{epoch:03d}"),
        save_freq= save_model_epoch*batch_per_epoch, #save per batch_num
        save_weights_only=False,
        save_best_only=False,
        monitor= 'val_loss', #'val_accuracy'
        mode= 'auto') #'max'

    history = model.fit(train_batch_data,
                        validation_data = eval_batch_data,
                        epochs=epoch_num, 
                        shuffle=True, 
                        callbacks=[model_checkpoint_callback]) #  # class_weight = weight_dict
    
    print('=== training process done!! ===')

    # save training history
    save_history_path = os.path.join(save_model_dir, 'motion_generation_transformer', 'training_history.pkl')
    joblib.dump(history.history, save_history_path)
    print('save model history to the path:', save_history_path)

    return history




def plot_training_history(history):
    plt.figure()
    plt.plot(history.history['loss'], color= 'b')
    plt.plot(history.history['val_loss'], color= 'r')
    plt.legend(['train', 'val'], loc='upper right')
    plt.title('Training loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()









if __name__ == '__main__':

    # ===== define parameters =====

    # define input data directory
    instrument_name = 'violin' 
    script_path = os.path.abspath(__file__)
    script_folder = os.path.abspath(os.path.join(script_path, os.pardir))
    parent_folder = os.path.abspath(os.path.join(script_folder, os.pardir))
    input_data_pkl_file = os.path.join(script_folder, instrument_name + "_training_data_pkg.pkl")
    print('=====')
    print('Input data from:', input_data_pkl_file)
    
    # define save model directory
    save_model_dir = script_folder
    print('Define save model folder:', save_model_dir)

    # input data parameters
    motion_normalize_range = 50
    audio_feature_size = 128
    label_feature_size = 10  
    time_frame_size =  640 # segement_time_frame_num

    # embedding parameters
    kernal_size = 11 # for embedding conv kernal
    audio_embed_dim = 64 # embedding size, must can be be divided as integer by num_heads
    label_embed_dim = 16
    num_heads = 4  # Number of attention heads
    rhand_output_dim = 12
    lhand_output_dim = 12
    head_output_dim = 9
    embed_dim = audio_embed_dim + label_embed_dim

    # model parameters
    dropout_rate = 0.1
    output_dim = 1
    num_layers_main_enc=3
    num_layers_branch_enc=3
    ff_dim = embed_dim * 4  # or *2, Hidden layer size in feed forward network inside transformer
    f_map = 16

    # loss weight parameters
    mse_loss_weight = 1
    mae_loss_weight = 0
    time_loss_weight = 2
    dim_loss_weight = 2


    rhand_loss_weight = 1
    lhand_loss_weight = 3
    head_loss_weight = 3

    # training parameters
    batch_size = 16
    epoch_num = 200
    save_model_epoch = 10
    validation_split = 0.1
    initial_learning_rate = 2e-3 

    print('define parameters done!')
    print('=====')


    # ===== load data =====

    train_data, eval_data = load_data_pkg_fn(
        input_data_pkl_file, 
        train_eval_split_ratio = int(1/validation_split),
        norm_range = motion_normalize_range
        )
    print('Input data from:', input_data_pkl_file)
    print('=====')


    # # # ===== build and train model =====
    print('Build transformer model:')
    motion_generation_model = build_transformer_model()

    training_history = training(
        motion_generation_model, 
        train_subset = train_data, 
        eval_subset= eval_data
        )

    
    # ===== check results =====
    print('Save model to folder:', os.path.join(save_model_dir, 'time_semantics_model'))
    plot_training_history(training_history)





    