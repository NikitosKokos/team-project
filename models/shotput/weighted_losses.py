from tensorflow.keras.losses import MeanSquaredError
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.saving import register_keras_serializable

# General fallback weighted_mse
@keras.saving.register_keras_serializable()
def weighted_mse(y_true, y_pred):
    # Default behavior: no specific weighting
    weights = K.ones_like(y_true)
    mse = K.mean(K.square(y_true - y_pred), axis=-1)
    return weights * mse

# Stage-specific loss functions
@keras.saving.register_keras_serializable()
def weighted_mse_stage1(y_true, y_pred):
    weights = K.switch(K.equal(y_true, 0), 2.0, K.switch(K.equal(y_true, 1), 1.5, 1.0))
    mse = K.mean(K.square(y_true - y_pred), axis=-1)
    return weights * mse

@keras.saving.register_keras_serializable()
def weighted_mse_stage2(y_true, y_pred):
    weights = K.switch(K.equal(y_true, 0), 1.8, K.switch(K.equal(y_true, 1), 1.2, 1.0))
    mse = K.mean(K.square(y_true - y_pred), axis=-1)
    return weights * mse

@keras.saving.register_keras_serializable()
def weighted_mse_stage3(y_true, y_pred):
    weights = K.switch(K.equal(y_true, 0), 2.5, K.switch(K.equal(y_true, 0.5), 1.5, 1.0))
    mse = K.mean(K.square(y_true - y_pred), axis=-1)
    return weights * mse

@keras.saving.register_keras_serializable()
def weighted_mse_stage4(y_true, y_pred):
    weights = K.switch(K.equal(y_true, 1), 2.0, K.switch(K.equal(y_true, 0), 1.8, 1.0))
    mse = K.mean(K.square(y_true - y_pred), axis=-1)
    return weights * mse

@keras.saving.register_keras_serializable()
def weighted_mse_stage5(y_true, y_pred):
    weights = K.switch(K.equal(y_true, 0), 3.0, K.switch(K.equal(y_true, 0.5), 1.5, 1.0))
    mse = K.mean(K.square(y_true - y_pred), axis=-1)
    return weights * mse
