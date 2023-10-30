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

np.set_printoptions(suppress=True, precision=3)

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
            data_arr = remove_nan_3d_array(data_arr, axis=1)
            data_arr = replace_nan_by_mean_3d_array(data_arr)
            # print('clean float array')
        
        load_data_pkg[key] = data_arr
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
        data_dict['beat_label'], 
        data_dict['downbeat_label'], 
        data_dict['phrase_label']
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
# focal loss
def focal_loss_from_probs(labels, preds):
    """Compute focal loss from probabilities. 
    ref.: https://github.com/artemmavrin/focal-loss/blob/master/src/focal_loss/_binary_focal_loss.py
    Parameters
    ----------
    labels : tensor-like
        Tensor of 0's and 1's: binary class labels.
    preds : tf.Tensor
        Estimated probabilities for the positive class.
    gamma : float
        Focusing parameter.
    alpha : float, weighting factor, [0,1]
        If not None, losses for the positive class will be scaled by this
        weight.
    Returns
    -------
    tf.Tensor
        The loss for each example.
    """
    # Predicted probabilities for the negative class
    alpha= 0.8 #0.25  #increase alpha: add weights on minority class
    gamma=2
    epsilon=1e-7
    mask=None
    reduce=True
    
    preds = preds + 1e-10 # avoid 0
    q = (1 - preds) # [B, T]
    labels_pos = tf.where(tf.greater(labels, 0), tf.ones_like(labels), labels)
    labels_neg = (1 - labels_pos)

    # Loss for the positive examples
    loss_pos = -alpha * (q**gamma) * tf.math.log(preds + epsilon) # [B, T]

    # Loss for the negative examples
    loss_neg = -(1-alpha) * (preds**gamma) * tf.math.log(q + epsilon) # [B, T]

    # Combine loss terms
    loss = labels_pos * loss_pos + labels_neg * loss_neg # [B, T]

    if mask is not None:
        loss *= mask

    return tf.reduce_mean(loss) if reduce else loss


# dice loss
def dice_loss_from_probs(labels, preds):
    '''
    labels, p, and mask with shape = [b, n] or [b, n, d]
    '''
    
    mask=None
    epsilon=1e-5
    axis=[1]
    
    preds = preds + 1e-10 # avoid 0
    if mask is not None:
        labels *= mask
        preds *= mask
    numerator = 2 * tf.reduce_sum(preds*labels, axis=axis, name='pq') # [b]
    denominator = tf.reduce_sum(preds**2, axis=axis, name='pp') + tf.reduce_sum(labels**2, axis=axis, name='qq') # [b]
    dice_coef = (numerator + epsilon) / (denominator + epsilon) # [b]
    dice_loss = 1 - dice_coef # [b]

    return tf.reduce_mean(dice_loss)

# define custom loss function
def custom_loss_fn(labels, preds):

    focal_loss_weight = 5
    dice_loss_weight = 1

    focal_weight = focal_loss_weight
    dice_weight = dice_loss_weight

    focal_loss = focal_loss_from_probs(labels, preds)
    dice_loss = dice_loss_from_probs(labels, preds)
    
    total_loss = (focal_weight*focal_loss) + (dice_weight*dice_loss)
    
    return total_loss


def load_model_fn(model_file):

        model = tf.keras.models.load_model(
            model_file, 
            custom_objects={"custom_loss_fn": custom_loss_fn}
            ) 
        model.summary()

        return model


def extract_id_item(id_list):
    '''
    get invidual items of id
    input: id list
    output: id item list {'song', 'perf', 'trial', 'seg'}
    '''

    new_list = []
    
    for item in id_list:

        id_split = item.split('_')
        file = id_split[0] + "_" + id_split[1] + "_" + id_split[2]
        serial = int(id_split[-1])

        id_dict = {
            'song': id_split[0],
            'perf': id_split[1],
            'trial': id_split[2],
            'seg': serial,
            'file': file
            }
        
        new_list.append(id_dict)
        
    return new_list
    

def find_file_names(id_item_list):
    '''
    get song name list from id item list
    input: id_item_list from fn 'extract_id_item': {'song', 'perf', 'trial', 'seg'}
    output: song_name_list
    '''

    unique_list = []

    file_name = [item['file'] for item in id_item_list]

    for name in file_name:
        if name not in unique_list:
            unique_list.append(name)
            
    return unique_list
    
    
def get_song_pred_dict(label_pkg, song_name_list):
    '''
    Get song pred pkg
    input: label_pkg {'pred_prob', 'true_ label', 'id'}
            song_name_list from func find_file_names
    output: song dict {song_names: {'pred_prob', 'true_label', 'id'}}
    '''

    # initiate new dict
    divide_label_dict = {name: [] for name in song_name_list}
    
    # get pred, label and len
    pred_list = label_pkg['pred_prob']
    label_list = label_pkg['true_label']
    
    # get label type keys
    pred_keys = list(label_pkg['pred_prob'].keys())
    label_keys = list(label_pkg['true_label'].keys())

    # get label len
    label_len = len(pred_list[pred_keys[0]])
    label_len = pred_list[pred_keys[0]].shape[0]
    print('label_len:', label_len)

    # loop over all labels
    for label_inx in range(label_len):
        
        pred_prob = {}
        true_label = {}
        
        # for each type of prediction
        for key in pred_keys:
            #print('label inx, keys:', label_inx, key)
            pred_prob[key] = pred_list[key][label_inx]
            true_label[key] = label_list[key][label_inx]
        
        id_name = label_pkg['id'][label_inx]
        file_name = id_name['file']

        label_item = {'pred_prob': pred_prob,
                    'true_label': true_label,
                    'id': id_name
                    }


        # search for all song names
        for song_inx in range(len(song_name_list)):
            song_name = song_name_list[song_inx]

            if song_name == file_name:
                divide_label_dict[song_name].append(label_item)

    # check songs and label len
    label_len_list = []

    for key in divide_label_dict.keys():
        label_len_list.append(len(divide_label_dict[key]))
        print('%s label len: %d' % (key, len(divide_label_dict[key])))

    total_label_len = np.sum(label_len_list)
    print('total label seg num:', total_label_len)
    print('song dict keys:', divide_label_dict[song_name][0].keys())
    
    return divide_label_dict


def find_prob_peak(
        pred_prob,
        threshold=0.5, 
        max_bpm=180,
        sampling_rate = 40
        ):

    peak_distance=sampling_rate * 60 / max_bpm
    
    peak_inx, _ = scipy.signal.find_peaks(pred_prob, 
                                        distance=peak_distance, 
                                        height=threshold)
    peak_value = [pred_prob[inx] for inx in peak_inx] 

    shape = len(pred_prob)
    prob_peak = np.zeros(shape)
    for pair_inx in range(len(peak_inx)):
        prob_peak[peak_inx[pair_inx]] = peak_value[pair_inx]
        
    return prob_peak


def prob_to_ones(input_list, threshold=0.5):
    '''
    convert model prediction prob into 1 or 0
    '''
    input_list = np.array(input_list)
    shape = input_list.shape

    print()

    new_arr = np.zeros(shape)
    new_arr[np.where(input_list >= threshold)] = 1
    
    return new_arr


def _fast_hit_windows(ref, est, window):
    '''Fast calculation of windowed hits for time events.
    Given two lists of event times ``ref`` and ``est``, and a
    tolerance window, computes a list of pairings
    ``(i, j)`` where ``|ref[i] - est[j]| <= window``.
    This is equivalent to, but more efficient than the following:
    >>> hit_ref, hit_est = np.where(np.abs(np.subtract.outer(ref, est))
    ...                             <= window)
    Parameters
    ----------
    ref : np.ndarray, shape=(n,)
        Array of reference values
    est : np.ndarray, shape=(m,)
        Array of estimated values
    window : float >= 0
        Size of the tolerance window
    Returns
    -------
    hit_ref : np.ndarray
    hit_est : np.ndarray
        indices such that ``|hit_ref[i] - hit_est[i]| <= window``
    '''

    ref = np.asarray(ref)
    est = np.asarray(est)
    ref_idx = np.argsort(ref)
    ref_sorted = ref[ref_idx]

    left_idx = np.searchsorted(ref_sorted, est - window, side='left')
    right_idx = np.searchsorted(ref_sorted, est + window, side='right')

    hit_ref, hit_est = [], []

    for j, (start, end) in enumerate(zip(left_idx, right_idx)):
        hit_ref.extend(ref_idx[start:end])
        hit_est.extend([j] * (end - start))

    return hit_ref, hit_est


def _bipartite_match(graph):
    """Find maximum cardinality matching of a bipartite graph (U,V,E).
    The input format is a dictionary mapping members of U to a list
    of their neighbors in V.
    The output is a dict M mapping members of V to their matches in U.
    Parameters
    ----------
    graph : dictionary : left-vertex -> list of right vertices
        The input bipartite graph.  Each edge need only be specified once.
    Returns
    -------
    matching : dictionary : right-vertex -> left vertex
        A maximal bipartite matching.
    """
    # Adapted from:
    #
    # Hopcroft-Karp bipartite max-cardinality matching and max independent set
    # David Eppstein, UC Irvine, 27 Apr 2002

    # initialize greedy matching (redundant, but faster than full search)
    matching = {}
    for u in graph:
        for v in graph[u]:
            if v not in matching:
                matching[v] = u
                break

    while True:
        # structure residual graph into layers
        # pred[u] gives the neighbor in the previous layer for u in U
        # preds[v] gives a list of neighbors in the previous layer for v in V
        # unmatched gives a list of unmatched vertices in final layer of V,
        # and is also used as a flag value for pred[u] when u is in the first
        # layer
        preds = {}
        unmatched = []
        pred = dict([(u, unmatched) for u in graph])
        for v in matching:
            del pred[matching[v]]
        layer = list(pred)

        # repeatedly extend layering structure by another pair of layers
        while layer and not unmatched:
            new_layer = {}
            for u in layer:
                for v in graph[u]:
                    if v not in preds:
                        new_layer.setdefault(v, []).append(u)
            layer = []
            for v in new_layer:
                preds[v] = new_layer[v]
                if v in matching:
                    layer.append(matching[v])
                    pred[matching[v]] = v
                else:
                    unmatched.append(v)

        # did we finish layering without finding any alternating paths?
        if not unmatched:
            unlayered = {}
            for u in graph:
                for v in graph[u]:
                    if v not in preds:
                        unlayered[v] = None
            return matching

        def recurse(v):
            """Recursively search backward through layers to find alternating
            paths.  recursion returns true if found path, false otherwise
            """
            if v in preds:
                L = preds[v]
                del preds[v]
                for u in L:
                    if u in pred:
                        pu = pred[u]
                        del pred[u]
                        if pu is unmatched or recurse(pu):
                            matching[v] = u
                            return True
            return False

        for v in unmatched:
            recurse(v)


def match_events(ref, est, window, distance=None):
    """Compute a maximum matching between reference and estimated event times,
    subject to a window constraint.
    Given two lists of event times ``ref`` and ``est``, we seek the largest set
    of correspondences ``(ref[i], est[j])`` such that
    ``distance(ref[i], est[j]) <= window``, and each
    ``ref[i]`` and ``est[j]`` is matched at most once.
    This is useful for computing precision/recall metrics in beat tracking,
    onset detection, and segmentation.
    Parameters
    ----------
    ref : np.ndarray, shape=(n,)
        Array of reference values
    est : np.ndarray, shape=(m,)
        Array of estimated values
    window : float > 0
        Size of the window.
    distance : function
        function that computes the outer distance of ref and est.
        By default uses ``|ref[i] - est[j]|``
    Returns
    -------
    matching : list of tuples
        A list of matched reference and event numbers.
        ``matching[i] == (i, j)`` where ``ref[i]`` matches ``est[j]``.
    """
    if distance is not None:
        # Compute the indices of feasible pairings
        hits = np.where(distance(ref, est) <= window)
    else:
        hits = _fast_hit_windows(ref, est, window)

    # Construct the graph input
    G = {}
    for ref_i, est_i in zip(*hits):
        if est_i not in G:
            G[est_i] = []
        G[est_i].append(ref_i)

    # Compute the maximum matching
    matching = sorted(_bipartite_match(G).items())

    return matching


def f_measure(precision, recall, beta=1.0):
    """Compute the f-measure from precision and recall scores.
    Parameters
    ----------
    precision : float in (0, 1]
        Precision
    recall : float in (0, 1]
        Recall
    beta : float > 0
        Weighting factor for f-measure
        (Default value = 1.0)
    Returns
    -------
    f_measure : float
        The weighted f-measure
    """

    if precision == 0 and recall == 0:
        return 0.0

    return (1 + beta**2)*precision*recall/((beta**2)*precision + recall)


def boundary_measure(reference_boundaries, estimated_boundaries, window=0.5, beta=1.0):
    """Boundary detection hit-rate.
    A hit is counted whenever an reference boundary is within ``window`` of a
    estimated boundary.  Note that each boundary is matched at most once: this
    is achieved by computing the size of a maximal matching between reference
    and estimated boundary points, subject to the window constraint.
    Examples
    --------
    >>> ref_intervals, _ = mir_eval.io.load_labeled_intervals('ref.lab')
    >>> est_intervals, _ = mir_eval.io.load_labeled_intervals('est.lab')
    >>> # With 0.5s windowing
    >>> P05, R05, F05 = mir_eval.segment.detection(ref_intervals,
    ...                                            est_intervals,
    ...                                            window=0.5)
    >>> # With 3s windowing
    >>> P3, R3, F3 = mir_eval.segment.detection(ref_intervals,
    ...                                         est_intervals,
    ...                                         window=3)
    >>> # Ignoring hits for the beginning and end of track
    >>> P, R, F = mir_eval.segment.detection(ref_intervals,
    ...                                      est_intervals,
    ...                                      window=0.5,
    ...                                      trim=True)
    Parameters
    ----------
    reference_intervals : np.ndarray, shape=(n, 2)
        reference segment intervals, in the format returned by
        :func:`mir_eval.io.load_intervals` or
        :func:`mir_eval.io.load_labeled_intervals`.
    estimated_intervals : np.ndarray, shape=(m, 2)
        estimated segment intervals, in the format returned by
        :func:`mir_eval.io.load_intervals` or
        :func:`mir_eval.io.load_labeled_intervals`.
    window : float > 0
        size of the window of 'correctness' around ground-truth beats
        (in seconds)
        (Default value = 0.5)
    beta : float > 0
        weighting constant for F-measure.
        (Default value = 1.0)
    trim : boolean
        if ``True``, the first and last boundary times are ignored.
        Typically, these denote start (0) and end-markers.
        (Default value = False)
    Returns
    -------
    precision : float
        precision of estimated predictions
    recall : float
        recall of reference reference boundaries
    f_measure : float
        F-measure (weighted harmonic mean of ``precision`` and ``recall``)
    """

    # If we have no boundaries, we get no score.
    if len(reference_boundaries) == 0 or len(estimated_boundaries) == 0:
        return 0.0, 0.0, 0.0

    matching = match_events(reference_boundaries,
                            estimated_boundaries,
                            window)

    precision = float(len(matching)) / len(estimated_boundaries)
    recall = float(len(matching)) / len(reference_boundaries)

    f_score = f_measure(precision, recall, beta=beta)

    return precision, recall, f_score


def get_pred_pkg(model, data_list, label_list, id_list):
    '''
    Get prediction package for per song
    input: model, data_list, label_list, id_list
    output: song dict {song_names: {'pred_prob', 'true_label', 'id'}}
    '''

    # get prediction
    prediction = model.predict(data_list)
    print('=====')
    print('Get model predictions:')
    print('pred types:', len(prediction))
    for inx in range(len(prediction)):
        print(
            'pred type ', inx, 
            'pred shape:', prediction[inx].shape, 
            'label shape:', label_list[inx].shape
            )

    # get id items
    id_item_list = extract_id_item(id_list)
    song_name_list = find_file_names(id_item_list)
    print('=====')
    print('Test song num:', len(song_name_list))
    for name in song_name_list:
        print(name)

    # combine pred, label, id into package
    comb_pkg = {
        'pred_prob': {},
        'true_label': {}, 
        'id': id_item_list
        }

    comb_pkg['pred_prob'] = {
        'beat': prediction[0],
        'downbeat': prediction[1],
        'phrase': prediction[2]
        }

    comb_pkg['true_label'] = {
        'beat': label_list[0],
        'downbeat': label_list[1],
        'phrase': label_list[2]
        }
    
    return comb_pkg



def convert_pred_pkg_to_time(
        pred_pkg,
        sampling_rate = 40
        ):
    
    result_dict = {}
    feature_key_list = eval_key_list
    
    # select threshold
    for key in feature_key_list:
        
        if key == 'beat':
            threshold = beat_threshold
            max_bpm = beat_max_bpm
        
        elif key == 'downbeat':
            threshold = downbeat_threshold
            max_bpm = downbeat_max_bpm
        
        print('label type:', key) 
        print('threshold:', threshold, 'max_bpm', max_bpm)
        
        # get flatten pred and label
        pred_1d = pred_pkg['pred_prob'][key].flatten()
        label_1d = pred_pkg['true_label'][key].flatten()

        pred_peak = find_prob_peak(
            pred_1d, 
            threshold=threshold, 
            max_bpm=max_bpm, 
            sampling_rate= sampling_rate
            )
        pred_onehot = prob_to_ones(pred_peak, threshold=threshold)
        print('label type:', key) 
    
        # from frame to time
        pred_frame, = np.where(np.array(pred_onehot).astype(int) == 1)
        label_frame, = np.where(np.array(label_1d).astype(int) == 1)
        # print('pred frame:', pred_frame)
        # print('label frame:', label_frame)

        pred_time = np.array(pred_frame) / sampling_rate
        label_time = np.array(label_frame) / sampling_rate
        # print('pred time:', pred_time)
        # print('label time:', label_time)
        
        print('label num: %d, predict num: %d' % (len(label_time), len(pred_time)))
    
    # save result as dict for each song
        result_dict[key] = {
            'label_time': label_time,
            'pred_time': pred_time,
            'label_frame_onehot': label_1d,
            'pred_frame_onehot': pred_onehot,
            'pred_frame_prob': label_1d,
            'pred_frame_peak': pred_peak
            }
        
    return result_dict



def calculate_score(time_result_pkg):
    '''
    calculate scores
    input: time_result_pkg
    output:
    score_dict_per_song song: {pred_feature: [presicion, recall, f1]}
    overall_score pred_feature: [presicion, recall, f1]
    '''

    score_list = {}
    feature_key_list = eval_key_list

    tolerance_list = {
            'beat': beat_window_size, 
            'downbeat': downbeat_window_size
            }

    # for each type of label
    # calculate PRF
    for key in feature_key_list:
        reference = time_result_pkg[key]['label_time']
        estimation = time_result_pkg[key]['pred_time']
        
        tolerance = tolerance_list[key]
        presicion, recall, f1 = boundary_measure(reference, estimation, window=tolerance)
        score_list[key] = [presicion, recall, f1]
        print(key, 'presicion: %.4f, recall: %.4f, F1: %.4f' %(presicion, recall, f1))

    return score_list





if __name__ == '__main__':

    # ===== define test data directory =====
    instrument_name = 'violin' 
    script_path = os.path.abspath(__file__)
    script_folder = os.path.abspath(os.path.join(script_path, os.pardir))
    parent_folder = os.path.abspath(os.path.join(script_folder, os.pardir))
    test_data_pkl_file = os.path.join(script_folder, instrument_name + "_testing_data_pkg.pkl")
    print('=====')
    print('Define test data:', test_data_pkl_file)


    # ===== define test model directory =====
    script_path = os.path.abspath(__file__)
    script_folder = os.path.abspath(os.path.join(script_path, os.pardir))

    # load pre-trained model
    test_model_file = os.path.join(script_folder, "pre_trained_model_time_semantics_violin")
    print('=====')
    print('Define test model:', test_model_file)


    # ===== define evaluation thresholds =====
    input_type = ['motion', 'audio']
    eval_key_list = ['beat', 'downbeat']
    data_sampling_rate = 40 
    beat_threshold = 0.5  
    downbeat_threshold = 0.3   
    phrase_threshold = 0.3 

    beat_window_size = 0.07 
    downbeat_window_size = 0.07 
    phrase_window_size = 0.7*2

    beat_max_bpm = 180
    downbeat_max_bpm = 90
    phrase_max_bpm = 20


    # ===== load testing data & model =====
    test_data = load_data_pkg_fn(test_data_pkl_file)
    test_data_list, test_label_list, test_id_list = get_data_label_id_list(test_data)
    print('Load testing data from:', test_data_pkl_file)

    test_model = load_model_fn(test_model_file)
    print('Load testing model from:', test_model_file)

    # ===== test =====
    pred_pkg = get_pred_pkg(
        test_model, 
        test_data_list, 
        test_label_list, 
        test_id_list)
    
    time_result_pkg = convert_pred_pkg_to_time(
        pred_pkg,
        sampling_rate = data_sampling_rate
        )
    
    song_scores = calculate_score(time_result_pkg)
    

    