import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow_addons.losses import PinballLoss
import neurokit2 as nk2
from __main__ import app
import numpy as np

class Model(object):

    def __init__(self):
        self.len = 1000
        self.lead = 12
        self.sr = 100
        self.esr = 500
        self.info_num = 2
    
    def process(self, ecg):
        cleaned_ecg = np.zeros((self.len, self.lead))
        for lead in range(self.lead):
            cleaned_ecg[:,lead] = nk2.signal.signal_resample(ecg[:,lead], 
                                                     desired_length = self.len,
                                                     sampling_rate = self.esr,
                                                     desired_sampling_rate = self.sr,
                                                     method = "FFT")
            cleaned_ecg[:, lead] = nk2.ecg.ecg_clean(cleaned_ecg[:,lead], sampling_rate = self.sr)
        return cleaned_ecg
    
    def predict(self, ecg, info):
        cleaned_ecg = self.process(ecg)
        cleaned_ecg = cleaned_ecg.reshape((1, self.len, self.lead))
        if info.shape == ():
            model = tf.keras.models.load_model(os.path.join(app.config["MODEL_PATH"], "pinball_no_info.hdf5"), compile=False)
            pred = model.predict(cleaned_ecg)
            info = np.zeros((1,self.info_num,1))
        else:
            model = tf.keras.models.load_model(os.path.join(app.config["MODEL_PATH"], "pinball_info.hdf5"), compile=False)
            info = info.reshape((1, self.info_num, 1))
            pred = model.predict([(cleaned_ecg, info)])
        return cleaned_ecg, info, pred
