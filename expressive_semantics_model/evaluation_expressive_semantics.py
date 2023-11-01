# supress numpy future warning
import warnings
warnings.filterwarnings('ignore')
import librosa, os, sys, pickle, mir_eval, joblib, datetime
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_nlp
import scipy.signal
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




def get_data_and_time_label_from_pkg(data_pkg):
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
            # print('clean float array')
        
        # convert string label to one hot label
        if key == 'dyn_label':
            new_data_arr = convert_str_to_one_hot_label(
            data_arr,
            label = ['ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff'],
            one_hot = [0,0,1,2,3,4,5,5])
        
        if key == 'arti_label':
            new_data_arr = convert_str_to_one_hot_label(
            data_arr,
            label = ['legato', 'staccato'],
            one_hot = [0,1])

        load_data_pkg[key] = new_data_arr
        print(key, load_data_pkg[key].shape)
        # print('data type:', load_data_pkg[key].dtype)

    print('---')  
    print('Check data value:')
    for key in load_data_pkg.keys():
        if key == 'motion_pos_data' or key == 'motion_vel_data' or key == 'audio_data':
            print(key, 'max: {:.2f}, min: {:.2f}, mean: {:.2f}'.format(
                np.nanmax(load_data_pkg[key]), 
                np.nanmin(load_data_pkg[key]),
                np.nanmean(np.absolute(load_data_pkg[key]))))

    del data_pkg
    return load_data_pkg



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


def get_data_label_id_list(data_dict):
    '''
    get data_list, label_list, id_list for testing data
    input: data_dict {key: data/label/id}
    output: data_list, label_list, id_list
    '''

    data_list = []

    print('test data type:')
    for name in input_type:

        if name == 'audio':
            data_list.append(data_dict['audio_data'])
            print('audio')
        elif name == 'motion':
            data_list.append(data_dict['motion_vel_data'])
            print('motion')

    label_list = [
        data_dict['dyn_label'], 
        data_dict['arti_label']
        ]

    id_list = data_dict['id']

    print('get data type num:', len(data_list))
    print('get label type num:', len(label_list))

    return data_list, label_list ,id_list


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


def load_data_pkg_fn(data_pkl_file): 

    input_data_pkg = joblib.load(data_pkl_file)
    print('Load data keys:', input_data_pkg[0].keys())
    print('Load data len:', len(input_data_pkg))

    input_data = get_data_and_time_label_from_pkg(input_data_pkg)
    del input_data_pkg

    return input_data



# define loss
def class_weighted_categorical_loss(one_hot_label, pred):

    weights = dyn_class_weight
    
    epsilon=1e-5
    pred = pred + epsilon

    pred = tf.math.log(pred)
    suppresed_logs = tf.math.multiply(pred, one_hot_label)
    loss =  -1*tf.math.multiply(suppresed_logs, weights)
    sum_loss = tf.math.reduce_mean(loss)
    
    return sum_loss



def load_model_fn(model_file):

        model = tf.keras.models.load_model(
            model_file,
            custom_objects={"class_weighted_categorical_loss": class_weighted_categorical_loss})
        
        model.summary()

        return model


def get_pred_pkg(
        model, 
        data_list, 
        label_list):
    '''
    get prediction from the model and ground truth label
    input: model, data_list, label_list
    output: dict{'dyn': {'pred', 'label'},
                'arti': {'pred', 'label'}}
    '''

    prediction = model.predict(data_list)
    # print('prediction', len(prediction))
    # for inx in range(len(prediction)):
    #     print(prediction[inx].shape)
    
    pred_pkg = {'dyn': {}, 'arti': {}}
    pred_pkg['dyn'] = {'pred': prediction[0], 'label': label_list[0]}
    pred_pkg['arti'] = {'pred': prediction[1], 'label': label_list[1]}

    return pred_pkg


def calculate_categorical_accuracy(pred, label):

    m = tf.keras.metrics.CategoricalAccuracy()
    m.update_state(pred, label)
    acc = m.result().numpy()

    return acc
    

def get_pkg_accuracy(pred_pkg):

    for key in pred_pkg.keys():
        acc_dict = {}
        key_acc = calculate_categorical_accuracy(
            pred_pkg[key]['pred'], 
            pred_pkg[key]['label'])
        
        acc_dict[key] = key_acc
        print(key, key_acc)

    return acc_dict
    
    




if __name__ == '__main__':

    # ===== define test data directory =====
    instrument_name = 'violin' 
    script_path = os.path.abspath(__file__)
    script_folder = os.path.abspath(os.path.join(script_path, os.pardir))
    parent_folder = os.path.abspath(os.path.join(script_folder, os.pardir))
    test_data_pkl_file = os.path.join(script_folder, instrument_name + "_testing_data_pkg.pkl")
    
    print('=====')
    print('Define test data from:', test_data_pkl_file)


    # ===== define test model directory =====
    script_path = os.path.abspath(__file__)
    script_folder = os.path.abspath(os.path.join(script_path, os.pardir))

    # load pre-trained model
    test_model_file = os.path.join(script_folder, "pre_trained_model_expression_semantics_violin")
    print('=====')
    print('Define test model from:', test_model_file)


    # ===== define evaluation parameters =====
    input_type = ['motion', 'audio']
    dyn_loss_weight = 1
    tech_loss_weight = 1
    dyn_class_weight = [5., 1., 5., 5., 1., 5.]



    # ===== load testing data & model =====
    test_data = load_data_pkg_fn(test_data_pkl_file)
    test_data_list, test_label_list, test_id_list = get_data_label_id_list(test_data)
    print('Load testing data from:', test_data_pkl_file)

    test_model = load_model_fn(test_model_file)
    print('Load testing model from:', test_model_file)

    # ===== evaluate prediction =====

    pred_pkg = get_pred_pkg(
        test_model,
        test_data_list,
        test_label_list)
    
    accuracy = get_pkg_accuracy(pred_pkg)


        
        
