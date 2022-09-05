#https://stackoverflow.com/questions/49846207/tensorflow-estimator-warm-start-from-and-model-dir

import tensorflow.compat.v1 as tf
import logging

class InitHook(tf.train.SessionRunHook):
    """initializes model from a checkpoint_path
    args:
        modelPath: full path to checkpoint
    """
    def __init__(self, checkpoint_dir):
        self.modelPath = checkpoint_dir
        self.initialized = False

    def begin(self):
        """
        Restore encoder parameters if a pre-trained encoder model is available and we haven't trained previously
        """
        if not self.initialized:
            log = logging.getLogger('tensorflow')
            checkpoint = tf.train.latest_checkpoint(self.modelPath)
            if checkpoint is None:
                log.info('No pre-trained model is available, training from scratch.')
            else:
                log.info('Pre-trained model {0} found in {1} - warmstarting.'.format(checkpoint, self.modelPath))
                tf.train.warm_start(checkpoint)
            self.initialized = True