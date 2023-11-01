# supress numpy future warning
import warnings
warnings.filterwarnings('ignore')
import librosa, os, sys, pickle, mir_eval, joblib, datetime, time 
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


def load_data_pkg_fn(data_pkl_file, train_eval_split_ratio):

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
    train_data = get_data_and_time_label_from_pkg(train_data_pkg)

    print('=====')
    print('get eval data:')
    eval_data = get_data_and_time_label_from_pkg(eval_data_pkg)

    del train_data_pkg
    del eval_data_pkg
    
    return train_data, eval_data



class DataGenerator_expression_semantics(tf.keras.utils.Sequence):
    """
    Custom data generator class for Digits dataset
    input: input data label pkg
    output: [input_data], [input_labels]
    """
    def __init__(self, data, batch_size=32):
        # self.audio_data = np.squeeze(data['audio_data'], axis = -1) # reduce 4-d data to 3-d
        # self.motion_data = np.squeeze(data['motion_vel_data'], axis = -1)
        self.audio_data = data['audio_data']
        self.motion_data = data['motion_vel_data']
        self.dyn_label = data['dyn_label']
        self.arti_label = data['arti_label']
        self.batch_size = batch_size
        
    def __len__(self):
        return int(np.math.floor(self.audio_data.shape[0] / self.batch_size))
    
    def __getitem__(self, index):
        """
        Returns a batch of data
        """       
        # Generate indexes of the batch
        ind_start = index*self.batch_size
        inx_end = (index+1)*self.batch_size
        
        batch_audio = self.audio_data[ind_start:inx_end, :, :]
        batch_motion = self.motion_data[ind_start:inx_end, :, :]
        batch_dyn_label = self.dyn_label[ind_start:inx_end, :]
        batch_arti_label = self.arti_label[ind_start:inx_end, :]
    
        # return index, ind_start, inx_end
        batch_data_list = []
        
        for data_type in input_type:
            if data_type == 'audio':
                data = batch_audio
            
            if data_type == 'motion':
                data = batch_motion

            batch_data_list.append(data)
        
        return batch_data_list, [batch_dyn_label, batch_arti_label]


class ConvEmbedding(layers.Layer):
    '''
    Define data embedding layer
    '''
    
    def __init__(self, num_hid, maxlen, kernal_size, stride=1):
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
        name = 'audio_input') # (time, feature)
    
    motion_inputs = layers.Input(
        shape=(time_frame_size, motion_feature_size), 
        name = 'motion_input') # (time, feature)
        
    # embedding
    audio_embedding_layer = ConvEmbedding(
        num_hid=audio_embed_dim, 
        maxlen=time_frame_size, 
        kernal_size=embed_kernal_size
        )
    
    motion_embedding_layer = ConvEmbedding(
        num_hid=motion_embed_dim, 
        maxlen=time_frame_size, 
        kernal_size=embed_kernal_size
        )

    audio_data_embedding = audio_embedding_layer(audio_inputs)
    motion_data_embedding = motion_embedding_layer(motion_inputs)

    # ===== audio encoder =====
    for en_inx in range(num_layers_main_enc):

        encoder_layer = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=ff_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            activation="relu",
            normalize_first=True,
            name='audio_encoder' + str(en_inx+1)
        )
        
        # for the first decoder layer
        if en_inx == 0:
            audio_branch = encoder_layer(audio_data_embedding)
        else:
            audio_branch = encoder_layer(audio_branch)
            
            
    # ===== motion encoder =====
    for en_inx in range(num_layers_main_enc):

        encoder_layer = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=ff_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            activation="relu",
            normalize_first=True,
            name='motion_encoder' + str(en_inx+1)
        )
        
        # for the first decoder layer
        if en_inx == 0:
            motion_branch = encoder_layer(motion_data_embedding)
        else:
            motion_branch = encoder_layer(motion_branch)        
    
        
    # ===== main branch =====
    # concat multiple data
    branch_list = []
    for data_type in input_type:
        if data_type == 'audio':
            branch = audio_branch
        elif data_type == 'motion':
            branch = motion_branch
        
        branch_list.append(branch)
        print('concate branch:', data_type)

    main_branch = tf.concat(branch_list, axis=-1, name='main_branch')  


    # ===== branch encoder =====
    # ===== dyn_encoder =====
    for en_inx in range(num_layers_branch_enc):
        
        encoder_layer = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=ff_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            activation="relu",
            normalize_first=True,
            name='dyn_encoder' + str(en_inx+1)
        )
        
        # for the first decoder layer
        if en_inx == 0:
            dyn_branch = encoder_layer(main_branch)
        else:
            dyn_branch = encoder_layer(dyn_branch)
        

    # dyn_output
    dyn_output = tf.keras.layers.Dense(f_map, name = 'dyn_dense_1')(dyn_branch)
    dyn_output = tf.nn.relu(dyn_output, name = 'dyn_relu_1')
    dyn_output = tf.keras.layers.Dense(dyn_output_size, name = 'dyn_dense_2')(dyn_output)
    dyn_output = tf.nn.softmax(dyn_output, axis=-1, name= 'dyn_softmax')
    dyn_output = tf.keras.layers.Reshape((time_frame_size, dyn_output_size), 
                                         input_shape=(time_frame_size, dyn_output_size), 
                                         name = 'dyn_output')(dyn_output)


    # ===== tech_decoder =====
    for en_inx in range(num_layers_branch_enc):
        
        encoder_layer = keras_nlp.layers.TransformerEncoder(
            intermediate_dim=ff_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            activation="relu",
            normalize_first=True,
            name='tech_encoder' + str(en_inx+1)
        )
        
        # for the first decoder layer
        if en_inx == 0:
            tech_branch = encoder_layer(main_branch)
        else:
            tech_branch = encoder_layer(tech_branch)

    # tech_output
    tech_output = tf.keras.layers.Dense(f_map, name = 'tech_dense_1')(tech_branch)
    tech_output = tf.nn.relu(tech_output, name = 'tech_relu_1')
    tech_output = tf.keras.layers.Dense(tech_output_size, name = 'tech_dense_2')(tech_output)
    tech_output = tf.nn.softmax(tech_output, axis=-1, name= 'tech_softmax')
    tech_output = tf.keras.layers.Reshape((time_frame_size, tech_output_size), 
                                          input_shape=(time_frame_size, tech_output_size), 
                                          name = 'tech_output')(tech_output)

    # build model
    input_list = []
    for data_type in input_type:
        if data_type == 'audio':
            inputs = audio_inputs
        if data_type == 'motion':
            inputs = motion_inputs
        
        input_list.append(inputs)

    model = keras.Model(
        input_list, 
        [dyn_output, tech_output], 
        name="transformer")    

    model.summary()

    return model




# define loss
def class_weighted_categorical_loss(one_hot_label, pred):
    '''
    from https://stackoverflow.com/questions/68330895/how-to-weight-classes-in-custom-categorical-crossentropy-loss-function
    calculate categorical loss for multiple classes
    input: 
    one-hot label [batch, time, class]
    predict prob [batch, time, class]
    '''

    weights = dyn_class_weight
    
    epsilon=1e-5
    pred = pred + epsilon

    pred = tf.math.log(pred)
    suppresed_logs = tf.math.multiply(pred, one_hot_label)
    loss =  -1*tf.math.multiply(suppresed_logs, weights)
    sum_loss = tf.math.reduce_mean(loss)
    
    return sum_loss



def training(model, train_subset, eval_subset):

    # learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = initial_learning_rate,
        decay_steps=10000, # batch num
        decay_rate=0.96,
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule) 

    model.compile(
        optimizer=optimizer,
        loss = {
            'dyn_output': class_weighted_categorical_loss,
            'tech_output': tf.keras.losses.CategoricalCrossentropy(from_logits=False)
            },

        loss_weights={
            'dyn_output': dyn_loss_weight, 
            'tech_output': tech_loss_weight
            }) 
        
    print('compile model done!')

    # get batch data from generator
    train_batch_data = DataGenerator_expression_semantics(train_subset, batch_size=batch_size)
    eval_batch_data = DataGenerator_expression_semantics(eval_subset, batch_size=batch_size)
    print('training data batch:', len(train_batch_data))
    print('eval data batch:', len(eval_batch_data))

    batch_per_epoch = len(train_batch_data)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath= os.path.join(save_model_dir, 'expression_semantics_transformer', "{epoch:03d}"),
        save_freq= save_model_epoch*batch_per_epoch, #save per n batch
        save_weights_only=False,
        save_best_only=False,
        monitor= 'val_loss', 
        mode= 'auto') 

    history = model.fit(
        train_batch_data,
        validation_data = eval_batch_data,
        epochs=epoch_num, 
        shuffle=True, 
        callbacks=[model_checkpoint_callback])
    
    print('=== training process done!! ===')
    
    # save training history
    save_history_path = os.path.join(save_model_dir, 'expression_semantics_transformer', 'training_history.pkl')
    joblib.dump(history.history, save_history_path)
    print('Save training history to:', save_history_path)

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
    input_type = ['motion', 'audio'] # 'audio', 'motion'
    audio_feature_size = 128
    motion_feature_size = 66
    dyn_output_size = 6
    tech_output_size = 2    
    time_frame_size =  640 # segement_time_frame_num

    # embedding parameters
    embed_kernal_size = 21 # for embedding conv kernal
    audio_embed_dim = 32 # embedding size, must can be be divided as integer by num_heads
    motion_embed_dim = 32
    num_heads = 4  # Number of attention heads
    embed_dim = audio_embed_dim + motion_embed_dim

    # model parameters
    dropout_rate = 0.1
    output_dim = 1
    num_layers_main_enc=3
    num_layers_branch_enc=3
    ff_dim = embed_dim * 4  # or *2, Hidden layer size in feed forward network inside transformer
    f_map = 16

    # loss weight parameters
    dyn_loss_weight = 1
    tech_loss_weight = 1
    dyn_class_weight = [5., 1., 5., 5., 1., 5.]

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
        train_eval_split_ratio = int(1/validation_split)
        )
    print('Input data from:', input_data_pkl_file)
    print('=====')


    # # ===== build and train model =====
    print('Build transformer model:')
    expression_semantic_model = build_transformer_model()

    training_history = training(
        expression_semantic_model, 
        train_subset = train_data, 
        eval_subset= eval_data
        )

    
    # ===== check results =====
    print('Save model to folder:', os.path.join(save_model_dir, 'time_semantics_model'))
    plot_training_history(training_history)





    