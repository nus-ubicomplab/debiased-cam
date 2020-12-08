'''

This script contains functions to train different CNN models (finetune or debias) 

- compute_class_weights: get the weights for each class label to balance the training.

- heatmapLoss: loss term for CAM faithfulness

- get_iterator: get the data iterator from annotation frames (pre-processing)

- batch_generator_finetune: generate the data batch for finetuned CNNs (pre-processing)

- batch_generator_debias: generate the data batch for debiased CNNs (pre-processing)

- batch_generator_siamese: generate the data batch for siamese CNNs (pre-processing)

- init_multitask_cnn: define the multi-task model architecture

- init_deabised_model: define the debiased-CAM training 

- init_siamese_model: define the siamese structure for debiased-CAM training

- train_finetune: train the finetuned CNNs 

- train_debias: train the debiased CNNs 

- train_siamese: train the debiased CNNs under the siamese structure

'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.utils import class_weight

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_control_flow_v2()

from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Lambda, Dense, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.xception import Xception, preprocess_input

base_model = None
regular_model = None
debiased_model = None

# for ImageNette
N_CLASSES = 10
num_train_samples = 9296
num_validation_samples = 3856

IMG_WIDTH_MAP = {'InceptionV3':299, 'VGG16':224, 'ResNet50':224, 'Xception':299}
CAM_WIDTH_MAP = {'InceptionV3':8, 'VGG16':14, 'ResNet50':7, 'Xception':10}
FREEZE_LAYER_MAP = {'InceptionV3':249, 'VGG16':15, 'ResNet50':165, 'Xception':115}
LAST_CONV_LAYER_MAP = {"InceptionV3":"mixed10", "VGG16":"block5_conv3", "ResNet50":"conv5_block3_add", "Xception":"block14_sepconv2_bn"}

def compute_class_weights(train_dir, ext='.JPEG'):
    categories = sorted(next(os.walk(train_dir))[1])
    targets = list()
    for i, category in enumerate(categories):
        cat_dir = os.path.join(train_dir, category)
        targets.extend([i for name in os.listdir(cat_dir) if name.endswith(ext)])
    return class_weight.compute_class_weight('balanced', np.arange(len(categories)), targets)

# run once as global var
class_weight = compute_class_weights("./datasets/imagenette/images/nobias/train") # used for re-balancing if data is class imbalanced

def mean_squared_error_2d(s_true, s_pred):
    # Define mse for 2 dimension space
    s_true = K.cast(s_true, s_pred.dtype)
    return K.mean(K.square(s_pred - s_true), axis=[1,2])

def differentiable_cam_loss():    
    def loss(s_true, s_pred):
        return mean_squared_error_2d(s_true, s_pred)
    return loss

"""
Modify Gram-CAM explainer as trainable and differentiable component in CNN.
Define the rule to obtain CAM output layer.
"""         
def Alpha_GAP(inputs):
    activation_maps, label_prediction, label_actual = inputs
    y_c = label_prediction * label_actual # get classification probability as one-hot vector; multiplication is Hadarmard operator
    # utility function to normalize a tensor by its L2 norm
    def _normalize(x):
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-8)
    gradients = _normalize(K.gradients(y_c, activation_maps)[0]) # To calculate the gradient of y^c by activation maps, the L2 norm is calculating 1/Z                          
    alpha_weights = K.mean(gradients[:,:,:,:], axis=(1,2)) # To calculate the alpha importance weights
    return alpha_weights

def GradCAM_Pool(inputs):
    activation_maps, alpha_weights = inputs
    cam = tf.einsum('ijkm,im->ijk', activation_maps, alpha_weights)     # Grad-CAM as matrix multiplication
    cam = tf.maximum(cam, 0)                                            # Relu activation
    cam = cam / ( K.max(cam, axis=(1,2), keepdims=True) + K.epsilon() ) # Min-max scaling
    return cam
    
def GradCAM_classic(model, label_actual, layer_name='mixed10'): # layer_name defined by MAP depends on base_cnn used
    label_prediction = model.layers[-1].output
    activation_maps = model.get_layer(layer_name).output # feature maps
    
    inputs = [activation_maps, label_prediction, label_actual]
    alpha_weights = Alpha_GAP(inputs)
    
    inputs = activation_maps, alpha_weights
    cam = GradCAM_Pool(inputs)
    return cam

def label_cam_loss(cam_weight=0):
    last_conv_layername = LAST_CONV_LAYER_MAP[base_model]
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, y_pred.dtype)
        label_loss = K.categorical_crossentropy(y_true, y_pred)
        # instead of defining the lambda layers inside model, defining the GradCAM as term in the labeling loss
        debiased_cam = GradCAM(debias_model, y_true, last_conv_layername) 
        regular_cam = GradCAM(regular_model, y_true, last_conv_layername+"_reg")
        cam_loss = mean_squared_error_2d(reg_cam, debias_cam) 
        return label_loss + cam_weight * cam_loss
    return loss

def get_iterator(frame_file, label_flag=0, batch_size=64, img_shape=(299, 299)):
    random.seed(1)
    datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)
    anno_frame = pd.read_csv(frame_file, dtype=str)
    datasplit = frame_file.split('.')[-2].split('_')[-1]
    biastype = frame_file.split('/')[-1].split('_')[0]

    # Flag controls the shuffle type: true for training and flase for testing
    shuffle_flag = True if datasplit == "train" else False
    # Flag controls the label type: labeling or bias regression
    if label_flag == 0:
        y_label = "label"
        class_mode = "categorical"
    elif label_flag == 1:
        y_label = "gaussian_kernel" if biastype != 'ct' else 'color_temp'
        class_mode = "raw"

    iterator = datagen.flow_from_dataframe(
        dataframe=anno_frame,
        directory=".",
        x_col="filename",
        y_col=y_label,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        seed=1,
        target_size=img_shape,
    )
    return iterator

# dataset generator used inside train_finetune()
def batch_generator_finetune(img_labels_file, batch_size, img_sz=299, heatmap_sz=8, data_dir="."):
    img_labels = get_iterator(img_labels_file, 0, batch_size, (img_sz, img_sz))
    img_blurlevels = get_iterator(img_labels_file, 1, batch_size, (img_sz, img_sz))

    while True:
        chunk_img_labels = img_labels.next()
        chunk_img_blurlevels = img_blurlevels.next()
        # Virtual CAMs are generated for a more consistent model structure between finetuned CNNs and debiased CNNs
        zero_cam = np.zeros((chunk_img_labels[0].shape[0], heatmap_sz, heatmap_sz))
        yield ([chunk_img_labels[0], chunk_img_labels[1]], [chunk_img_labels[1], zero_cam, np.array(chunk_img_blurlevels[1]).astype(float)])

# dataset generator used inside train_debias()
def batch_generator_debias(img_labels_file, biased_img_labels_file, batch_size, img_sz=299, nobias_data_dir=".", bias_data_dir="."):
    img_labels = get_iterator(img_labels_file, 0, batch_size, (img_sz, img_sz))
    biased_img_labels = get_iterator(biased_img_labels_file, 0, batch_size, (img_sz, img_sz))
    biased_img_blurlevels = get_iterator(biased_img_labels_file, 1, batch_size, (img_sz, img_sz))

    while True:
        chunk_img_labels = img_labels.next()
        chunk_biased_img_labels = biased_img_labels.next()
        chunk_biased_img_blurlevels = biased_img_blurlevels.next()
        blurlevel = np.array(chunk_biased_img_blurlevels[1]).astype(float)

        with graph.as_default():    
            tf.compat.v1.keras.backend.set_session(sess)
            # Regular CAMs are used to debias the training.
            _, unbiased_cam, _ = regular_model.predict([chunk_img_labels[0], chunk_img_labels[1]], batch_size=batch_size)

        # return inputs and outputs of multi-input, multi-task model (with siamese parallel RegularCNN)
        # yield returns value via the function at each loop iteration
        yield ([chunk_biased_img_labels[0], chunk_biased_img_labels[1]], [chunk_biased_img_labels[1], unbiased_cam, blurlevel])

# dataset generator used inside train_siamese()
def batch_generator_siamese(img_labels_file, biased_img_labels_file, batch_size, cam_width=8, img_width=299):
    img_labels = get_iterator(img_labels_file, 0, batch_size, (img_width, img_width)) # rows of [image pixels, image_classification_label]
    biased_img_labels = get_iterator(biased_img_labels_file, 0, batch_size, (img_width, img_width)) # rows of [image pixels, image_classification_label]
    biased_img_blurlevels =  get_iterator(biased_img_labels_file, 1, batch_size, (img_width, img_width)) # rows of [image pixels, image_biaslevel_label]

    while True:
        chunk_img_labels= img_labels.next() # [image pixels, image_classification_label]
        chunk_biased_img_labels = biased_img_labels.next() # [image pixels, image_classification_label]
        chunk_biased_img_blurlevels = biased_img_blurlevels.next() # [image pixels, image_biaslevel_label]
        blurlevel = np.array(chunk_biased_img_blurlevels[1]).astype(float)

        
        with graph.as_default():    
            tf.compat.v1.keras.backend.set_session(sess)
            # predict from RegularCNN (as self-supervised oracle), but only keep CAM explanation, i.e., Unbiased-CAM
            _, unbiased_cam, _ = regular_model.predict([chunk_img_labels[0], chunk_img_labels[1]], batch_size=batch_size) 

        # return inputs and outputs of multi-input, multi-task model (with siamese parallel RegularCNN)
        # yield returns value via the function at each loop iteration
        yield ([chunk_biased_img_labels[0], chunk_biased_img_labels[1], chunk_img_labels[0], chunk_img_labels[1]], # model inputs v2
               [chunk_biased_img_labels[1], zero_cam, blurlevel, chunk_img_labels[1], zero_cam, blurlevel]) # model outputs v2

"""
To Define the Multi-task training framework
"""
def init_multitask_cnn(base_model="InceptionV3"):
    freeze_layer = FREEZE_LAYER_MAP[base_model]
    img_sz = IMG_WIDTH_MAP[base_model]

    if base_model == "InceptionV3":
        base_cnn_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_sz, img_sz, 3))
    elif base_model == "VGG16":
        base_cnn_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_sz, img_sz, 3))  
    elif base_model == "ResNet50":
        base_cnn_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_sz, img_sz, 3))  
    elif base_model == "Xception":    
        base_cnn_model = Xception(weights='imagenet', include_top=False, input_shape=(img_sz, img_sz, 3))
    
    last_conv = base_cnn_model.output
    img_embedding_layer = GlobalAveragePooling2D(name='img_embedding_layer')(last_conv)
    label_out = Dense(N_CLASSES, activation='softmax', name="label_out")(img_embedding_layer)
    label_input = Input(shape=(N_CLASSES,), name="label_input") 
    
    """
    Define the tensor shape.
    """
    def GAPalpha_output_shape(input_shapes):
        output_shape = (input_shapes[0][0], input_shapes[0][3])
        return output_shape

    def GradCAM_output_shape(input_shapes):
        output_shape = input_shapes[0][0:3]
        return output_shape

    """
    Lambda layer used to define the heatmap as a output.
    """
    alpha_weights = Lambda(Alpha_GAP, GAPalpha_output_shape, name="alpha_weights")([last_conv, label_out, label_input])
    cam_out = Lambda(GradCAM_Pool, GradCAM_output_shape, name="cam_out")([last_conv, alpha_weights]) 

    """
    Define the bias-level learning relevant layers.
    """
    biaslevel_fc = Dense(2048, activation='relu', name='biaslevel_fc')(img_embedding_layer)
    biaslevel_dropout = Dropout(0.5, name='biaslevel_dropout')(biaslevel_fc)
    biaslevel_out = Dense(1, name='biaslevel_out')(biaslevel_dropout)              

    model = Model(
        inputs=[base_cnn_model.input, label_input], # 2 inputs: image, y ground truth (or prediction??)
        outputs=[label_out, cam_out, biaslevel_out] # 3 outputs
    )
    
    # Freezing of the shallow layers for a more efficient training
    for layer in model.layers[:freeze_layer]:
        layer.trainable = False
    for layer in model.layers[freeze_layer:]:
        layer.trainable = True
    return model 

# This model only has DebiasedCNN, RegularCNN is only defined as an oracle to provide unbiased-CAMs 
def init_deabised_model(weights_dir, bias_level, bias_type="blur", base_model="InceptionV3"):
    reg_weight = weights_dir+'/{}_nobias_reg.hdf5'.format(base_model)
    bias_weight = weights_dir+'/{}_{}_finetune_bias{}.hdf5'.format(base_model, bias_type, bias_level)

    global regular_model, graph, sess
    sess = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        tf.compat.v1.keras.backend.set_session(sess)
        regular_model = init_multitask_cnn()
        debias_model = init_multitask_cnn()
        
        if load_weight == True:
            debias_model.load_weights(bias_weight, by_name=False)
            regular_model.load_weights(reg_weight, by_name=False)
        
        # Setting RegCNN non-trainable
        for layer in regular_model.layers:
            layer._name = layer._name + str("_reg")
            layer.trainable = False
        return debias_model

# This model has RegularCNN and DebiasedCNN in parallel such that they can be synchronized with same input image
# RegularCNN portion is not trainable (all frozen)
def init_siamese_model(weights_dir, bias_level, load_weight=True, bias_type="blur", base_model="InceptionV3"):
    reg_weight = weights_dir+'/{}_nobias_reg.hdf5'.format(base_model)
    bias_weight = weights_dir+'/{}_{}_finetune_bias{}.hdf5'.format(base_model, bias_type, bias_level)
    
    global regular_model, debias_model, graph, sess
    sess = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()   
    with graph.as_default():
        tf.compat.v1.keras.backend.set_session(sess)
        regular_model = init_multitask_cnn()
        debias_model = init_multitask_cnn()
        
        if load_weight == True:
            debias_model.load_weights(bias_weight, by_name=False)
            regular_model.load_weights(reg_weight, by_name=False)
        
        # Setting RegularCNN non-trainable
        for layer in regular_model.layers:
            layer._name = layer._name + str("_reg") # append names with _reg for reference by GradCAM_classic
            layer.trainable = False
        
        # Defining the structure for siamese CNN model
        siamese_model = Model(
            inputs  = [debias_model.inputs[0], debias_model.inputs[1], 
                        regular_model.inputs[0], regular_model.inputs[1]], 
            outputs = [debias_model.outputs[0], debias_model.outputs[1], debias_model.outputs[2], 
                       regular_model.outputs[0], regular_model.outputs[1], regular_model.outputs[2]]
        )
        return siamese_model


def train_finetune(biastype, biaslevel, weights_dir, batch_size, num_epochs, base_model, initial_epoch=0):
    backend = 'tf' if K.backend() == 'tensorflow' else 'th'
    img_width = IMG_WIDTH_MAP[base_model]
    cam_width = CAM_WIDTH_MAP[base_model]
    databias = "nobias" if biaslevel == '0' else biastype+'_'+biaslevel

    train_generator = batch_generator_finetune("./datasets/imagenette/{}_train.csv".format(databias), batch_size, img_width, cam_width)
    val_generator = batch_generator_finetune("./datasets/imagenette/{}_val.csv".format(databias), batch_size, img_width, cam_width)
    
    # Ratios based on hyper-parameter tuning.
    biaslevel_weight = 2e-5 if biaslevel == 'multibias' else 0
    adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, amsgrad=False) 
    model = init_multitask_cnn(base_model)
    model.compile(
        loss={
            'label_out':'categorical_crossentropy', 
            'cam_out':differentiable_cam_loss(), 
            'biaslevel_out':'mean_squared_error'
        }, 
        loss_weights={
            'label_out':1.0, 
            'cam_out':0.0, 
            'biaslevel_out':biaslevel_weight
        },
        optimizer=adam,
    )

    model_name = "/{}_{}_finetune_{}.hdf5".format(base_model, biastype, databias) if databias != "nobias" else "/{}_nobias_reg.hdf5".format(base_model)
    if initial_epoch != 0: model.load_weights(weights_dir + model_name, by_name=True)
    weights_path = weights_dir + model_name
    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(patience=6)
    tensorboard = TensorBoard(log_dir="logs/{}_{}_finetune_{}".format(base_model, biastype, databias))

    model.fit(
        train_generator,
        steps_per_epoch=num_train_samples//batch_size,
        validation_data=val_generator,
        validation_steps=num_validation_samples//batch_size,
        epochs=num_epochs,
        verbose=1,
        callbacks=[checkpoint, earlystopping, tensorboard],
        class_weight=[class_weight, None, None],
        initial_epoch=initial_epoch
    )
    if K.backend() == 'tensorflow': K.clear_session()

def train_debias(biastype, biaslevel, weights_dir, batch_size, num_epochs, base_model, initial_epoch=0):
    backend = 'tf' if K.backend() == 'tensorflow' else 'th'
    img_width = IMG_WIDTH_MAP[base_model]
    databias = "nobias" if biaslevel == '0' else biastype+'_'+biaslevel

    train_generator = batch_generator_debias("./datasets/imagenette/nobias_train.csv", "./datasets/imagenette/{}_train.csv".format(databias), batch_size, img_width)
    val_generator = batch_generator_debias("./datasets/imagenette/nobias_val.csv", "./datasets/imagenette/{}_val.csv".format(databias), batch_size, img_width)

    # Ratios based on hyper-parameter tuning.
    cam_loss_weight = 100.0
    biaslevel_weight = 2e-3 if biaslevel == 'multibias' else 0
    adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model = init_siamese_model(weights_dir, biaslevel)
    model.compile(
        loss={
            'label_out':'categorical_crossentropy', 
            'cam_out':differentiable_cam_loss(), 
            'biaslevel_out':'mean_squared_error'
        }, 
        loss_weights={
            'label_out':1.0, 
            'cam_out':cam_loss_weight, 
            'biaslevel_out':biaslevel_weight
        },
        optimizer=adam, 
    )

    if initial_epoch != 0: model.load_weights(weights_dir + '/{}_{}_debias_{}.hdf5'.format(base_model, biastype, databias), by_name=True)
    weights_path = weights_dir + "/{}_{}_debias_{}.hdf5".format(base_model, biastype, databias)
    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(patience=6)
    tensorboard = TensorBoard(log_dir="logs/{}_{}_debias_{}".format(base_model, biastype, databias))

    model.fit(
        train_generator,
        steps_per_epoch=num_train_samples//batch_size,
        validation_data=val_generator,
        validation_steps=num_validation_samples//batch_size,
        epochs=num_epochs,
        verbose=1,
        callbacks=[checkpoint, earlystopping, tensorboard],
        class_weight=[class_weight, None, None],
        initial_epoch=initial_epoch
    )

    if K.backend() == 'tensorflow': K.clear_session()

def train_siamese(biastype, biaslevel, weights_dir, batch_size, num_epochs, initial_epoch=0):
    backend = 'tf' if K.backend() == 'tensorflow' else 'th'
    cam_width = CAM_WIDTH_MAP[base_model]
    img_width = IMG_WIDTH_MAP[base_model]
    databias = "nobias" if biaslevel == '0' else biastype+'_'+biaslevel
    
    train_generator = batch_generator_siamese("./datasets/imagenette/nobias_train.csv", "./datasets/imagenette/{}_train.csv".format(databias), batch_size, img_width)
    val_generator = batch_generator_siamese("./datasets/imagenette/nobias_val.csv", "./datasets/imagenette/{}_val.csv".format(databias), batch_size, img_width)

    # Ratios based on hyper-parameter tuning.
    cam_loss_weight = 100.0 
    biaslevel_weight = 2e-3 if biaslevel == 'multibias' else 0
    adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, amsgrad=False) 
    model = init_siamese_model(weights_dir, biaslevel)
    
    # After training, the model is of the siamese structure. We only take the debias branch for evaluation 
    model.compile(
        loss = {
            'label_out':label_cam_loss(cam_loss_weight), 
            'cam_out':differentiable_cam_loss(), 
            'biaslevel_out':'mean_squared_error', 
            'label_out_reg':'categorical_crossentropy',
            'cam_out_reg':differentiable_cam_loss(), 
            'biaslevel_out_reg':'mean_squared_error', 
        }, 
        loss_weights = {
            'label_out':1.0, 
            'cam_out':0, 
            'biaslevel_out':biaslevel_weight, 
            'label_out_reg':0.0,
            'cam_out_reg':0,
            'biaslevel_out_reg':0,
        },
        optimizer = adam, 
    )

    if initial_epoch != 0: model.load_weights(weights_dir + '/{}_{}_siamese_{}.hdf5'.format(base_model, biastype, databias), by_name=True)    
    weights_path = siamese_weights_dir + '/{}_{}_siamese_{}.hdf5'.format(base_model, biastype, databias)
    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True) 
    earlystopping = EarlyStopping(patience=6)
    tensorboard = TensorBoard(log_dir="logs/{}_{}_siamese_{}".format(base_model, biastype, databias))

    model.fit(
        train_generator,
        steps_per_epoch=num_train_samples//batch_size,
        validation_data=val_generator,
        validation_steps=num_validation_samples//batch_size,
        epochs=num_epochs,
        verbose=1,
        callbacks=[checkpoint, earlystopping, tensorboard],
        class_weight=[class_weight, None, None, None, None, None],
        initial_epoch=initial_epoch
    )

    if K.backend() == 'tensorflow': K.clear_session()