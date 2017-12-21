import tensorflow as tf
from conf import conf
from keras.layers import (
        Conv2D, BatchNormalization, Input, Activation, Dense, Reshape,
        Add,
)
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Model, load_model
from keras.callbacks import TensorBoard
from keras.regularizers import l2
import os
import logging
from app_log import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

SIZE = conf['SIZE']
L2_EPSILON = conf['L2_EPSILON']

REGULARIZERS = {
    'kernel_regularizer': l2(L2_EPSILON),
    'bias_regularizer': l2(L2_EPSILON),
}


def residual_block(input_, node_name):
    with tf.name_scope(node_name):
        conv1 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', **REGULARIZERS)(input_)
        batch1 = BatchNormalization()(conv1)
        relu = Activation('relu')(batch1)
        conv2 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', **REGULARIZERS)(relu)
        batch2 = BatchNormalization()(conv2)
        add = Add()([batch2, input_])
        out = Activation('relu')(add)
    return out


def loss(y_true, y_pred):
    mse = K.mean(K.square(y_pred - y_true), axis=-1)
    categorical_crossentropy = K.categorical_crossentropy(y_true, y_pred)
    return mse + categorical_crossentropy


def build_model(name):
    with tf.name_scope('input'):
        _input = Input(shape=(SIZE, SIZE, 17))
        conv1 = Conv2D(filters=256, kernel_size=(3, 3), strides=1,
                data_format='channels_last', **REGULARIZERS)(_input)
        batch1 = BatchNormalization()(conv1)
        relu = Activation('relu')(batch1)


    tower_input = relu
    with tf.name_scope('tower'):
        for i in range(conf['N_RESIDUAL_BLOCKS']):
            tower_output = residual_block(tower_input, node_name="residual_%s" % i)
            tower_input = tower_output



    with tf.name_scope('policy'):
        policy_conv = Conv2D(filters=2, kernel_size=(1, 1), strides=1, **REGULARIZERS)(tower_output)
        policy_batch = BatchNormalization()(policy_conv)
        policy_relu = Activation('relu')(policy_batch)

        shape = policy_relu._keras_shape
        policy_shape = (shape[1] * shape[2] * shape[3], )
        policy_reshape = Reshape(target_shape=policy_shape)(policy_relu)
        policy_out = Dense(SIZE*SIZE + 1, activation='softmax', name="policy_out", **REGULARIZERS)(policy_reshape)

    with tf.name_scope('value'):
        value_conv = Conv2D(filters=2, kernel_size=(1, 1), strides=1, **REGULARIZERS)(tower_output)
        value_batch = BatchNormalization()(value_conv)
        value_relu = Activation('relu')(value_batch)
        shape = value_relu._keras_shape
        value_shape = (shape[1] * shape[2] * shape[3], )
        value_reshape = Reshape(target_shape=value_shape)(value_relu)
        value_hidden = Dense(256, activation='relu', **REGULARIZERS)(value_reshape)
        value_out = Dense(1, activation='tanh', name="value_out", **REGULARIZERS)(value_hidden)

    model = Model(inputs=[_input], outputs=[policy_out, value_out], name=name)
    sgd = SGD(lr=1e-2, momentum = 0.9)
    model.compile(sgd, loss=loss)
    return model


def create_initial_model(name, self_play=True):
    full_filename = os.path.join(conf['MODEL_DIR'], name) + ".h5"
    if os.path.isfile(full_filename):
        model = load_model(full_filename, custom_objects={'loss': loss})
        return model

    model = build_model(name)

    # Save graph in tensorboard. This graph has the name scopes making it look
    # good in tensorboard, the loaded models will not have the scopes.
    tf_callback = TensorBoard(log_dir=os.path.join(conf['LOG_DIR'], name),
            histogram_freq=0, batch_size=1, write_graph=True, write_grads=False)
    tf_callback.set_model(model)
    tf_callback.on_epoch_end(0)
    tf_callback.on_train_end(0)

    if self_play:
        from self_play import self_play
        self_play(model, n_games=conf['N_GAMES'], mcts_simulations=conf['MCTS_SIMULATIONS'])
    model.save(full_filename)
    best_filename = os.path.join(conf['MODEL_DIR'], 'best_model.h5')
    model.save(best_filename)
    return model


def load_latest_model():
    index = -1
    model_filename = None
    for filename in os.listdir(conf['MODEL_DIR']):
        try:
            name = filename.split('.')[0] # remove .h5
            i = int(name.split('_')[-1]) #may throw exception here
            if i > index:
                model_filename = filename
                index = i
        except:
            continue

    logger.debug("Loading latest model %s...", model_filename)
    model = load_model(os.path.join(conf['MODEL_DIR'], model_filename), custom_objects={'loss': loss})
    logger.debug("Loaded latest model %s", model.name)
    if model_filename.split('.')[0] != model.name:
        logger.warning(">>>>>>>> Inconsistent model name, should check!!! <<<<<<<<<")
    return model


def load_best_model():
    logger.debug("Loading best model...")
    best_model_path = os.path.join(conf['MODEL_DIR'], conf['BEST_MODEL'])
    if os.path.isfile(best_model_path):
        model = load_model(best_model_path, custom_objects={'loss': loss})
    else:
        logger.warning("Found no best model. Initializing new model...")
        model = create_initial_model(name="model_1", self_play=False)
    logger.debug("Loaded best model %s", model.name)
    return model


def load_model_by_name(name):
    logger.debug("Loading latest model %s ...", name)
    model = load_model(os.path.join(conf['MODEL_DIR'], name), custom_objects={'loss': loss})
    return model