import os
from utils import init_directories, clean_up_empty
import tensorflow as tf
from numpy import Inf
from keras.utils import multi_gpu_model
from conf import conf
import logging
from data_generator import DataGenerator, get_training_desc
from model import load_latest_model, loss, SGD
from app_log import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

#  Continuous training implementation

def main():
    init_directories()
    clean_up_empty()
    GPUs = conf['GPUs']
    EPOCHS_PER_SAVE = conf['EPOCHS_PER_SAVE']
    BATCH_SIZE = conf['TRAIN_BATCH_SIZE']
    NUM_WORKERS = conf['NUM_WORKERS']
    SIZE = conf['SIZE']
    n_gpu = len(GPUs)
    if n_gpu <= 1:
        raise EnvironmentError("Number of GPU need > 1 for multi-gpus training")

    logger.info("STARTING TRAINING PHASE with %s GPUs", len(GPUs))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUs).strip('[').strip(']').strip(' ')

    with tf.device('/cpu:0'):
        model = load_latest_model()

    base_name, index = model.name.split('_')
    best_loss = Inf

    pmodel = multi_gpu_model(model, gpus=n_gpu)
    opt = SGD(lr=1e-2, momentum=0.9)
    pmodel.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

    params = {'dim': (SIZE, SIZE, 17),
              'batch_size': BATCH_SIZE * n_gpu,
              'shuffle': True}
    while True:
        new_name = "_".join([base_name, str(int(index) + 1)]) + ".h5"
        partition, labels = get_training_desc()
        training_generator = DataGenerator(partition['train'], labels, **params)
        validation_generator = DataGenerator(partition['validation'], labels, **params)
        callbacks_list = []

        pmodel.fit_generator(generator=training_generator,
                             validation_data=validation_generator,
                             use_multiprocessing=True,
                             workers=NUM_WORKERS, epochs=EPOCHS_PER_SAVE,
                             callbacks=callbacks_list)

        curr_loss = model.evaluate_generator(generator=validation_generator)
        if curr_loss < best_loss:
            best_loss = curr_loss
            model.name = new_name.split('.')[0]
            model.save(os.path.join(conf['MODEL_DIR'], new_name))
            base_name, index = model.name.split('_')
            logger.info("Save New model %s", new_name)

if __name__ == "__main__":
    main()
