#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 13:28:16 2022

@author: palirezazadeh
"""
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


from PIL import Image
from albumentations import *
from skimage.transform import resize

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score

from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
from keras.models import load_model, Model

import tensorflow as tf
from keras import backend as K

from keras import regularizers
# from arcface_layer import ArcFace
from margin import ArcFace
# from layers import *
# from metrics import *

from sklearn.manifold import TSNE
from bioinfokit.visuz import cluster
from sklearn.cluster import KMeans

SHAPE = (224, 224, 3)
BATCH_SIZE = 24
EPOCHS = 100
N_SPLITS = 5
SEED = 1881
TRAIN_TEST_RATIO = 0.2

BASE_DIR     = "../data/BreaKHis_v1/histology_slides/breast/"
DATASET_MODE = ["40X", "100X", "200X", "400X"]

class BREAKHIST_DATASET:
    
    """
    input_shape           --> TUPLE.wanted image size
    batch_size            --> INT.yielding data size for every iteration
    orders                --> LIST.which images will be used. max=len(all_images). it can be used for K-fold(CV).
    base_dir              --> STR.the DIR which is include "benign" and "malignant" dirs.
    dataset_mode          --> STR. Which type of images will be used: "40X", "100X", "200X", "400X".
    seed                  --> INT. This allow to dataset generator to more reproduciable and it ensures that x and y are shuffled with compatible.
    augment               --> BOOL. Augment data or not.
    train_test_ratio      --> How much of data will be used as test set.
    ---------
    GENERAL_CLASSES       --> LIST.["benign", "malignant"]
    BENIGN_SUB_CLASSES    --> LIST.["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma"]
    MALIGNANT_SUB_CLASSES --> LIST.["ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"]
    """
    
    def __init__(self, input_shape, batch_size, orders, base_dir, dataset_mode, seed, train_test_ratio, augment=True):
        self.SHAPE                 = input_shape
        self.BATCH_SIZE            = batch_size
        self.arr                   = orders
        self.DATASET_MODE          = dataset_mode
        self.SEED                  = seed
        self.TT_RATIO              = train_test_ratio
        self.AUG                   = augment
        
        self.BASE_DIR              = base_dir
        self.GENERAL_CLASSES       = ["benign", "malignant"]
        self.BENIGN_SUB_CLASSES    = ["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma"]
        self.MALIGNANT_SUB_CLASSES = ["ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"]
        
        
    def get_paths_n_labels(self):

        x      = []
        label = []
        
        for data_mode in self.DATASET_MODE:
            for ix1, a in enumerate(self.GENERAL_CLASSES):
                if ix1 == 0:
                    for ix2, b in enumerate(self.BENIGN_SUB_CLASSES):
                        path1 = self.BASE_DIR+a+"/SOB/"+b
                        for c in os.listdir(path1):
                            path2 = path1+"/"+c+"/"+data_mode
                            for img_name in os.listdir(path2):
                                path3 = path2+"/"+img_name

                                # x
                                img_path = path3 #np.array(Image.open(path3), dtype=np.float16)

                                # y
                                main_targets = np.zeros((2), dtype=np.float32) # BENIGN OR MALIGNANT
                                main_targets[ix1] = 1.

                                # Store the values
                                x.append(img_path)
                                label.append(main_targets)


                if ix1 == 1:
                    for ix2, b in enumerate(self.MALIGNANT_SUB_CLASSES):
                        path1 = self.BASE_DIR+a+"/SOB/"+b
                        for c in os.listdir(path1):
                            path2 = path1+"/"+c+"/"+data_mode
                            for img_name in os.listdir(path2):
                                path3 = path2+"/"+img_name

                                # x
                                img_path = path3  #np.array(Image.open(path3), dtype=np.float16)

                                # y
                                main_targets = np.zeros((2), dtype=np.float32) # BENIGN OR MALIGNANT
                                main_targets[ix1] = 1.

                                # Store the values
                                x.append(img_path)
                                label.append(main_targets)
                           
        return x, label
    
    def __len__(self):
        return len(self.get_paths_n_labels()[0])
    
    def get_img(self, img_path):
        img = Image.open(img_path)
        return np.array(img)
    
    def augmenting(self, img):
        if self.AUG:
            augment = Compose([VerticalFlip(p=0.5),
                               HorizontalFlip(p=0.5),
                               RandomBrightnessContrast(p=0.3),
                               ShiftScaleRotate(p=0.5, shift_limit=0.2, scale_limit=0.2, rotate_limit=20)])  
        else:
            augment = Compose([])  

        img = augment(image=img)['image']
        return img
    
    
    def resize_and_normalize(self, img):
        img = resize(img, self.SHAPE)
        return img
    
    def get_shuffled_data(self):
        img_paths, labels = self.get_paths_n_labels()

        np.random.seed(self.SEED) 
        np.random.shuffle(img_paths)
        
        np.random.seed(self.SEED) 
        np.random.shuffle(labels)
        
        return img_paths, labels
    
    def split_train_test(self, get):  # get=={"train","test"}
        img_paths, labels = self.get_shuffled_data()
        x_train, x_test, y_train, y_test = train_test_split(img_paths, labels, test_size=self.TT_RATIO, random_state=self.SEED)
        
        if get=='train':
            return x_train, y_train
        
        elif get=='test':
            return x_test, y_test
    
    def data_generator(self):
        img_paths, labels = self.split_train_test(get="train")
        
        while True:
            x = np.empty((self.BATCH_SIZE,)+self.SHAPE, dtype=np.float32)
            y = np.empty((self.BATCH_SIZE, 2), dtype=np.float32)

            batch = np.random.choice(self.arr, self.BATCH_SIZE)

            for ix, id_ in enumerate(batch):
                # x
                img_path = img_paths[id_]
                img = self.get_img(img_path)
                img = self.augmenting(img)
                img = self.resize_and_normalize(img)
                  
                # y 
                label = labels[id_]
             
                # Store the values    
                x[ix] = img
                y[ix] = label

            yield x, y
            
            
dataset = BREAKHIST_DATASET(SHAPE, 1, range(4), BASE_DIR, DATASET_MODE, SEED, TRAIN_TEST_RATIO, augment=True)

# credits: https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
def recall(y_true, y_pred):
    """
    Recall metric.
    
    Only computes a batch-wise average of recall.
    
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.
    
    Only computes a batch-wise average of precision.
    
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precisionx = precision(y_true, y_pred)
    recallx = recall(y_true, y_pred)
    return 2*((precisionx*recallx)/(precisionx+recallx+K.epsilon()))


class SGDRScheduler(Callback):
    '''Cosine annealing learning rate scheduler with periodic restarts.
    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)
        

# copied from https://github.com/kobiso/CBAM-keras/blob/master/models/attention_module.py
def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]
    
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1,1,channel)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1,1,channel)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    
    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7
    
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature
    
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)	
    assert cbam_feature._keras_shape[-1] == 1
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
        
    return multiply([input_feature, cbam_feature])

# copied from https://gist.github.com/mjdietzx/5319e42637ed7ef095d430cb5c5e8c64
def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    y = Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
    y = BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    y = add([shortcut, y])
    y = LeakyReLU()(y)

    return y

af = ArcFace()
losses = [af.loss]


def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)

def create_model():
    # af_layer = ArcFace(output_dim=2, class_num=2, margin=0.5, scale=64.)
    dropRate = 0.3
    # label = Input(shape=(2,))
    init = Input(SHAPE)
    x = Conv2D(32, (3, 3), activation=None, padding='same')(init) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), activation=None, padding='same')(x) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x1 = MaxPooling2D((2,2))(x)
    
    x = Conv2D(64, (3, 3), activation=None, padding='same')(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = cbam_block(x)
    x = residual_block(x, 64)
    x2 = MaxPooling2D((2,2))(x)
    
    x = Conv2D(128, (3, 3), activation=None, padding='same')(x2)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = cbam_block(x)
    x = residual_block(x, 128)
    x3 = MaxPooling2D((2,2))(x)
    
    ginp1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x1)
    ginp2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(x2)
    ginp3 = UpSampling2D(size=(8, 8), interpolation='bilinear')(x3)
    
    hypercolumn = Concatenate()([ginp1, ginp2, ginp3]) 
    gap = GlobalAveragePooling2D()(hypercolumn)

    x = Dense(256, activation=None)(gap)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropRate)(x)
    
    x = Dense(256, activation=None)(x)
    x = BatchNormalization()(x)
    x  = Activation('relu')(x)
    
    y = Dense(2, activation='softmax')(x)        
       
    model = Model(init, y)
    return model

model = create_model()

kf = KFold(n_splits=N_SPLITS, random_state=SEED, shuffle=True)

for ix, (train_index, test_index) in enumerate(kf.split(range(len(dataset.split_train_test("train")[0])))):
                                               
    tg = BREAKHIST_DATASET(SHAPE, BATCH_SIZE, train_index, BASE_DIR, DATASET_MODE, SEED, TRAIN_TEST_RATIO, augment=True)
    vg = BREAKHIST_DATASET(SHAPE, BATCH_SIZE, test_index , BASE_DIR, DATASET_MODE, SEED, TRAIN_TEST_RATIO, augment=False)
        
    schedule = SGDRScheduler(min_lr=1e-6,
                              max_lr=1e-3,
                              steps_per_epoch=np.ceil(EPOCHS/BATCH_SIZE),
                              lr_decay=0.9,
                              cycle_length=10,
                              mult_factor=2.)

    model.compile(loss=losses, optimizer=Adam(lr=1e-3), metrics=['acc'])

    model_ckpt = "BREAKHIST_FOLD_"+str(ix)+".h5"
    callbacks = [ModelCheckpoint(model_ckpt, monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=False),
                  TensorBoard(log_dir='./log_'+str(ix), update_freq='epoch'), 
                  schedule] 
                                               
    model.fit_generator(tg.data_generator(),
                        steps_per_epoch=len(train_index)//BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=2,
                        validation_data=vg.data_generator(),
                        validation_steps=len(test_index)//BATCH_SIZE,
                        callbacks=callbacks)
    

def get_test_data():
    gen = BREAKHIST_DATASET(SHAPE, BATCH_SIZE, range(1), BASE_DIR, DATASET_MODE, SEED, TRAIN_TEST_RATIO, augment=False).split_train_test("test")
                       
    x = np.empty((len(gen[0]),)+SHAPE, dtype=np.float32)
    y = np.empty((len(gen[1]), 2), dtype=np.float32)
    
    for ix, path in tqdm(enumerate(gen[0])):
        img = np.array(Image.open(gen[0][ix]))
        img = resize(img, SHAPE)

        label = gen[1][ix]

        x[ix] = img
        y[ix] = label
        
    return x, y
x, y = get_test_data()



# Threshold predictions with THRESH_VAL
def threshold_arr(array):
    # Get all value from array
    # Compare calue with THRESH_VAL 
    # IF value >= THRESH_VAL. round to 1
    # ELSE. round to 0
    new_arr = []
    for ix, val in enumerate(array):
        loc = np.array(val).argmax(axis=0)
        k = list(np.zeros((len(val)), dtype=np.float32))
        k[loc]=1
        new_arr.append(k)
        
    return np.array(new_arr, dtype=np.float32)

models = []
for i in range(5):
    model = load_model("BREAKHIST_FOLD_{}.h5".format(i), custom_objects={'loss': losses})
    print(model.evaluate(x, y, verbose=0))
    models.append(model)


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

#     Citiation
#     ---------
#     http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

#     """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                      horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                      horizontalalignment="center",
                      color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig("CML.jpg",bbox_inches = 'tight', dpi=300)
    plt.show()

F=[]
Pr=[]
Re=[]
Ac=[]

y_preds1 = threshold_arr(models[0].predict([x,y], verbose=0))
y_preds2 = threshold_arr(models[1].predict([x,y], verbose=0))
y_preds3 = threshold_arr(models[2].predict([x,y], verbose=0))
y_preds4 = threshold_arr(models[3].predict([x,y], verbose=0))
y_preds5 = threshold_arr(models[4].predict([x,y], verbose=0))

#############Fold 1 ######################################
results1 = precision_recall_fscore_support(y, y_preds1 ,average='macro')
acc1 = accuracy_score(y, y_preds1)
Ac.append(acc1)
Pr.append(results1[0])
F.append(results1[2])
Re.append(results1[1])

##################Fold 2 ################################
results2 = precision_recall_fscore_support(y, y_preds2 ,average='macro')
acc2 = accuracy_score(y, y_preds2)
Ac.append(acc2)
Pr.append(results2[0])
F.append(results2[2])
Re.append(results2[1])

######################Fold 3 ##############################
results3 = precision_recall_fscore_support(y, y_preds3 ,average='macro')
acc3 = accuracy_score(y, y_preds3)
Ac.append(acc3)
Pr.append(results3[0])
F.append(results3[2])
Re.append(results3[1])

#######################Fold 4#############################
results4 = precision_recall_fscore_support(y, y_preds4 ,average='macro')
acc4 = accuracy_score(y, y_preds4)
Ac.append(acc4)
Pr.append(results4[0])
F.append(results4[2])
Re.append(results4[1])

##########################Fold 5 ###########################
results5 = precision_recall_fscore_support(y, y_preds5 ,average='macro')
acc5 = accuracy_score(y, y_preds5)
Ac.append(acc5)
Pr.append(results5[0])
F.append(results5[2])
Re.append(results5[1])

###############################################
print("Accuracy: {}, std: {}".format(np.mean(Ac), np.std(Ac)))
print("Persion: {}, std: {}".format(np.mean(Pr), np.std(Pr)))
print("Recall: {}, std: {}".format(np.mean(Re), np.std(Re)))
print("F-score: {}, std: {}".format(np.mean(F), np.std(F)))




cnf_matrix1 = confusion_matrix(y.argmax(axis=1), y_preds1.argmax(axis=1))
cnf_matrix2 = confusion_matrix(y.argmax(axis=1), y_preds2.argmax(axis=1))
cnf_matrix3 = confusion_matrix(y.argmax(axis=1), y_preds3.argmax(axis=1))
cnf_matrix4 = confusion_matrix(y.argmax(axis=1), y_preds4.argmax(axis=1))
cnf_matrix5 = confusion_matrix(y.argmax(axis=1), y_preds5.argmax(axis=1))

cnf_matrix = cnf_matrix1 + cnf_matrix2 + cnf_matrix3 + cnf_matrix4 + cnf_matrix5

plot_confusion_matrix(cm           = cnf_matrix, 
                      normalize    = False,
                      target_names = ['BENIGN', 'MALIGNANT'],
                      title        = "CAAM BAAM")

import time
for i in range(5): 
    img = np.array(Image.open(BASE_DIR+"benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-009.png"))
    x = resize(img, SHAPE)
    x = x.reshape((1,) + x.shape) 
    start = time.time()
    prediction = models[3].predict(x, batch_size=1)
    finish = time.time()
    print(threshold_arr(prediction))
    print((finish-start)*1000,"ms")
    print("***")


############################Feature Space ############################

modelX = Model(inputs=models[3].input[0], outputs=models[3].layers[-3].output)
embedded_features = modelX.predict(x, verbose=1)
tsne = TSNE(n_components=2).fit_transform(embedded_features)

kmeans = KMeans(n_clusters=2)
kmeans.fit(tsne)
y_kmeans = kmeans.predict(tsne)
scatter=plt.scatter(tsne[:, 0], tsne[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.legend(handles=scatter.legend_elements()[0], labels=['BENIGN', 'MALIGNANT'],loc='lower right')
plt.savefig("DAM.jpg",bbox_inches = 'tight', dpi=300)
##################################Roc Curve ##############################################
from sklearn import metrics

models = []
model = load_model("BREAKHIST_FOLD_4.h5", custom_objects={'f1': f1, 'precision': precision, 'recall': recall, 'DAM': DAM})
models.append(model)
model = load_model("BREAKHIST_FOLD_4-A.h5", custom_objects={'f1': f1, 'precision': precision, 'recall': recall, 'SphereFace': SphereFace})
models.append(model)
model = load_model("BREAKHIST_FOLD_4-CosFace.h5", custom_objects={'f1': f1, 'precision': precision, 'recall': recall, 'CosFace': CosFace})
models.append(model)
model = load_model("BREAKHIST_FOLD_4-R.h5", custom_objects={'f1': f1, 'precision': precision, 'recall': recall, 'ArcFace': ArcFace})
models.append(model)
model = load_model("BREAKHIST_FOLD_4-CM.h5", custom_objects={'f1': f1, 'precision': precision, 'recall': recall, 'DAM': DAM})
models.append(model)

y_preds1 = models[0].predict([x,y])[:, 1]
y_preds2 = models[1].predict([x,y])[:, 1]
y_preds3 = models[2].predict([x,y])[:, 1]
y_preds4 = models[3].predict([x,y])[:, 1]
y_preds5 = models[4].predict([x,y])[:, 1]

plt.figure(0).clf()
plt.title('Fold 5')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

fpr, tpr, _ = metrics.roc_curve(y.argmax(axis=1), y_preds2)
auc = round(metrics.roc_auc_score(y.argmax(axis=1), y_preds2), 4)
plt.plot(fpr,tpr,label="Softmax, AUC="+str(auc))

fpr, tpr, _ = metrics.roc_curve(y.argmax(axis=1), y_preds3)
auc = round(metrics.roc_auc_score(y.argmax(axis=1), y_preds3), 4)
plt.plot(fpr,tpr,label="A-Softmax, AUC="+str(auc))

fpr, tpr, _ = metrics.roc_curve(y.argmax(axis=1), y_preds4)
auc = round(metrics.roc_auc_score(y.argmax(axis=1), y_preds4), 4)
plt.plot(fpr,tpr,label="CosFace, AUC="+str(auc))

fpr, tpr, _ = metrics.roc_curve(y.argmax(axis=1), y_preds1)
auc = round(metrics.roc_auc_score(y.argmax(axis=1), y_preds1), 4)
plt.plot(fpr,tpr,label="ArcFace, AUC="+str(auc))

fpr, tpr, _ = metrics.roc_curve(y.argmax(axis=1), y_preds5)
auc = round(metrics.roc_auc_score(y.argmax(axis=1), y_preds5), 4)
plt.plot(fpr,tpr,label="CAAM, AUC="+str(auc))

plt.xlim(-0.01, 0.3)
plt.ylim(0.7, 1.01)
plt.legend()
plt.savefig("fold5.jpg",bbox_inches = 'tight', dpi=300)

####################################### loss ##############################
import matplotlib.pyplot as plt
import tensorflow as tf

lst = list(range(1,101))
aa = []
for e in tf.train.summary_iterator('events.out.tfevents.1661744712.09CLIPCWOR003'):
    for v in e.summary.value:
        if v.tag == 'acc':
            aa.append(v.simple_value*100)

bb = []
for e in tf.train.summary_iterator('events.out.tfevents.1661744712.09CLIPCWOR003'):
    for v in e.summary.value:
        if v.tag == 'val_acc':
            bb.append(v.simple_value*100)

plt.figure(0).clf()
plt.title('DAM')
plt.ylabel('Accuracy(%)')
plt.xlabel('Epochs')
plt.plot(lst,aa,label="train")
plt.plot(lst,bb,label="val")
plt.legend()
plt.savefig("damAcc.jpg",bbox_inches = 'tight', dpi=300)
loss: 0.2634 - precision: 0.8805 - recall: 0.8805 - f1: 0.8805 - acc: 0.8805 - val_loss: 0.3951 - val_precision: 0.8558 - val_recall: 0.8558 - val_f1: 0.8558 - val_acc: 0.8558

