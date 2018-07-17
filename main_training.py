import os
from utils import init_directories, clean_up_empty
import tensorflow as tf
from numpy import Inf
from keras.utils import multi_gpu_model
from conf import conf
import logging
from data_generator import DataGenerator, get_KGS_training_desc, get_training_desc
from model import load_latest_model, loss, SGD, load_model_by_name
from scpy import sync_model
from keras.callbacks import ReduceLROnPlateau
import atexit
from app_log import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

#  Continuous training implementation

model = None


def save_backup_model():
    if model is not None:
        model.save(os.path.join(conf['MODEL_DIR'], "exit_backup.h5"))


def rename_model(model_file_name, new_model_name):
    m = load_model_by_name(model_file_name)
    m.name = new_model_name
    m.save(os.path.join(conf['MODEL_DIR'], new_model_name + ".h5"))
    print("Done")


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

    global model
    with tf.device('/cpu:0'):
        model = load_latest_model()

    base_name, index = model.name.split('_')
    smallest_loss = Inf

    pmodel = multi_gpu_model(model, gpus=n_gpu)
    opt = SGD(lr=1e-2, momentum=0.9)
    pmodel.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

    params = {'dim': (SIZE, SIZE, 17),
              'batch_size': BATCH_SIZE * n_gpu,
              'shuffle': True}
    while True:
        new_name = "_".join([base_name, str(int(index) + 1)]) + ".h5"
        partition = get_KGS_training_desc()  # get_training_desc()
        training_generator = DataGenerator(partition['train'], None, **params)
        validation_generator = DataGenerator(partition['validation'], None, **params)
        reduce_lr = ReduceLROnPlateau(monitor='policy_out_acc', factor=0.1, patience=3, verbose=1, mode='auto', min_lr=0)

        callbacks_list = [reduce_lr]

        EPOCHS_PER_BACKUP = conf['EPOCHS_PER_BACKUP']
        cycle = EPOCHS_PER_SAVE//EPOCHS_PER_BACKUP
        for i in range(cycle):
            logger.info("CYCLE {}/{}".format(i+1, cycle))
            pmodel.fit_generator(generator=training_generator,
                                 # validation_data=validation_generator,
                                 use_multiprocessing=True,
                                 workers=NUM_WORKERS, epochs=EPOCHS_PER_BACKUP, verbose=1,
                                 callbacks=callbacks_list)
            model.save(os.path.join(conf['MODEL_DIR'], "backup.h5"))
            logger.info('Auto save model backup.h5')

        logger.info("Validating model")
        curr_loss = model.evaluate_generator(generator=validation_generator) # ['loss', 'policy_out_loss', 'value_out_loss']
        logger.info('Validation result: %s', curr_loss)
        if curr_loss[0] < smallest_loss:
            logger.info("Model improves. Validation loss from {} to {}. Save this model as {}".format(smallest_loss, curr_loss[0], new_name))
            smallest_loss = curr_loss[0]
            model.name = new_name.split('.')[0]
            model.save(os.path.join(conf['MODEL_DIR'], new_name))
            logger.info("Saved new model %s", new_name)
            sync_model(new_name)  # copy to other self-play servers
            base_name, index = model.name.split('_')


atexit.register(save_backup_model)
if __name__ == "__main__":
    main()
