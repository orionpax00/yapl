import tensorflow as tf
import torch 

import yalp

class BasicModel:
    def __init__(self, is_sequential=True):
        self.is_sequential = is_sequential
        self.backend = yalp.backend

    def dispatch(self):

        return self._model()

    def _model(self):
        if self.backend == 'tf':
            if self.is_sequential:
                #Simple binary classification model for 1D input
                model = tf.keras.Sequential()
                model.add(tf.keras.Input(shape=yalp.config.INPUT_SHAPE))

                if yalp.config.DATATYPE == 'img':
                    model.add(tf.keras.layers.Conv2D(32, 3, activation="relu"))
                    model.add(tf.keras.layers.Conv2D(64, 3, activation="relu"))
                    model.add(tf.keras.layers.MaxPooling2D(3))
                    model.add(tf.keras.layers.Flatten())

                if yalp.config.DATATYPE == 'text':
                    #TODO
                
                if yalp.config.DATATYPE == 'tabular':
                    model.add(tf.keras.layers.Dense(32))
                    model.add(tf.keras.layers.Dense(64))
                    model.add(tf.keras.layers.Dense(128))

                if yalp.config.PROBLEM_TYPE == "classification":
                    model.add(tf.keras.layers.Dense(yalp.config.NUM_CLASSES, activation='sigmoid'))
                
                elif yalp.config.PROBLEM_TYPE == "regression":
                    model.add(tf.keras.layers.Dense(yalp.config.NUM_CLASSES))
                
                return model