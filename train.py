#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:57:18 2019

@author: nuoxu
"""
import cfg
import os
from east.preprocess import *
from east.label import *
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

import cfg
from east.network import East
from east.losses import quad_loss
from east.data_generator import gen

if __name__ == '__main__':
    preprocess()
    process_label()
    east = East()
    east_network = east.east_network()
    east_network.summary()
    east_network.compile(loss=quad_loss, optimizer=Adam(lr=cfg.lr,
                                                        # clipvalue=cfg.clipvalue,
                                                        decay=cfg.decay))
    if cfg.load_weights and os.path.exists(cfg.saved_model_weights_file_path):
        east_network.load_weights(cfg.saved_model_weights_file_path)
    
    east_network.fit_generator(generator=gen(),
                               steps_per_epoch=cfg.steps_per_epoch,
                               epochs=cfg.epoch_num,
                               validation_data=gen(is_val=True),
                               validation_steps=cfg.validation_steps,
                               verbose=1,
                               initial_epoch=cfg.initial_epoch,
                               callbacks=[
                                   EarlyStopping(patience=cfg.patience, verbose=1),
                                   ModelCheckpoint(filepath=cfg.model_weights_path,
                                                   save_best_only=True,
                                                   save_weights_only=True,
                                                   verbose=1)])
    east_network.save(cfg.saved_model_file_path)
    east_network.save_weights(cfg.saved_model_weights_file_path)