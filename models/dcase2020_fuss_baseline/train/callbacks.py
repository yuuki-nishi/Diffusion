import tensorflow as tf
from . import metrics as imported_metrics
class BatchLogCallback(tf.keras.callbacks.Callback):
    def __init__(self, params):
        #super(tf.keras.callbacks.Callback,self).__init__()
        self.train_summary_writer = tf.summary.create_file_writer(params["model_dir"]+"on_train_batch_end")
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(epoch, keys))
        with train_summary_writer.as_default():
        
            tf.summary.scalar(name = "sisnr_separated",data=tf.reduce_mean(sisnr_separated),step=batch)
            tf.summary.scalar(name = "sisnr_mixture",data=tf.reduce_mean(sisnr_mixture),step=batch)
            tf.summary.scalar(name = "sisnr_improvement",data=tf.reduce_mean(sisnr_separated - sisnr_mixture),step=batch)
            
