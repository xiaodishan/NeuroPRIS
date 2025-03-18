import argparse
import os
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import keras
from abc import ABC

SEQ_MAX_LENGTH = 150

class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule, ABC):
    def __init__(self, d_model, warmup_steps=50, **kwargs):
        super(CustomSchedule, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class OrthogonalRegularizer(keras.layers.Layer):
    def __init__(self, num_features, l2reg=0.001, **kwargs):
        super(OrthogonalRegularizer, self).__init__(**kwargs)
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def get_config(self):
        config = super(OrthogonalRegularizer, self).get_config().copy()
        config.update({
            "num_features": self.num_features,
            "l2reg": self.l2reg,
        })
        return config

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

class Conv_bn(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(Conv_bn, self).__init__(**kwargs)
        self.filters = filters
        self.conv_1d = layers.Conv1D(filters, kernel_size=1, padding='valid')
        self.conv_1d.supports_masking = True
        self.batch_normalization = layers.BatchNormalization(momentum=0.0)
        self.activation = layers.Activation('relu')
        self.supports_masking = True

    def get_config(self):
        config = super(Conv_bn, self).get_config().copy()
        config.update({
            "filters": self.filters
        })
        return config

    def __call__(self, inputs):
        x = self.conv_1d(inputs)
        x = self.batch_normalization(x)
        outputs = self.activation(x)
        return outputs

class Dense_bn(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(Dense_bn, self).__init__(**kwargs)
        self.filters = filters
        self.dense = layers.Dense(filters)
        self.batch_normalization = layers.BatchNormalization(momentum=0.0)
        self.activation = layers.Activation('relu')
        self.supports_masking = True

    def get_config(self):
        config = super(Dense_bn, self).get_config().copy()
        config.update({
            'filters': self.filters
        })
        return config

    def __call__(self, inputs):
        x = self.dense(inputs)
        x = self.batch_normalization(x)
        outputs = self.activation(x)
        return outputs

class Dense_Layer(keras.layers.Layer):
    def __init__(self, filters, dff, **kwargs):
        super(Dense_Layer, self).__init__(**kwargs)
        self.filters = filters
        self.dense = layers.Dense(dff, activation='relu')
        self.dense2 = layers.Dense(filters)
        self.layer_normalization = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(0.1)
        self.supports_masking = True

    def get_config(self):
        config = super(Dense_Layer, self).get_config().copy()
        config.update({
            'filters': self.filters
        })
        return config

    def __call__(self, inputs):
        x = self.dense(inputs)
        x = self.dense2(x)
        x = self.dropout(x)

        inputs_projection = layers.Dense(self.filters)(inputs)

        outputs = self.layer_normalization(inputs_projection + x)
        return outputs

class Tnet(keras.layers.Layer):
    def __init__(self, num_features, **kwargs):
        super(Tnet, self).__init__(**kwargs)
        self.num_features = num_features
        self.bias = keras.initializers.Constant(np.eye(num_features).flatten())
        self.reg = OrthogonalRegularizer(num_features)
        self.conv_bn_1 = Conv_bn(32)
        self.conv_bn_2 = Conv_bn(64)
        self.conv_bn_3 = Conv_bn(512)
        self.max_pooling = layers.GlobalMaxPooling1D()
        self.dense_bn_1 = Dense_bn(256)
        self.dense_bn_2 = Dense_bn(128)
        self.dense = layers.Dense(num_features * num_features,
                                  kernel_initializer="zeros",
                                  bias_initializer=self.bias,
                                  activity_regularizer=self.reg)
        self.reshape = layers.Reshape((num_features, num_features))
        self.dot = layers.Dot(axes=(2, 1))

    def get_config(self):
        config = super(Tnet, self).get_config().copy()
        config.update({
            'num_features': self.num_features
        })
        return config

    def __call__(self, inputs):
        x = self.conv_bn_1(inputs)
        x = self.conv_bn_2(x)
        x = self.conv_bn_3(x)
        x = self.max_pooling(x)
        x = self.dense_bn_1(x)
        x = self.dense_bn_2(x)
        x = self.dense(x)
        feat_T = self.reshape(x)
        outputs = self.dot([inputs, feat_T])
        return outputs

class NeuroPRIS:
    def __init__(self, num_points=1426, EPOCHS=20, dataname='', learning_rate=None, lstm_units=48):
        self.NUM_POINTS = num_points
        self.NUM_CLASSES = 2
        self.EPOCHS = EPOCHS
        self.myModel = None
        self.seq_length = SEQ_MAX_LENGTH
        self.modelname = f'{dataname}'
        self.learning_rate = learning_rate or 0.001
        self.lstm_units = lstm_units

    def model_build(self, return_seq=False):
        sequences_inputs = keras.Input(shape=self.seq_length, name='sequences_inputs')
        secshape_inputs = keras.Input(shape=self.seq_length, name='secshape_inputs')
        coords_3d_inputs = keras.Input(shape=(self.NUM_POINTS, 3), name='coords_3d_inputs')

        x1 = Tnet(3)(coords_3d_inputs)
        x1 = Conv_bn(32)(x1)
        x1 = Conv_bn(32)(x1)
        x1 = Tnet(32)(x1)
        x1 = Conv_bn(32)(x1)
        x1 = Conv_bn(64)(x1)
        x1 = Conv_bn(128)(x1)
        x1 = layers.GlobalMaxPooling1D()(x1)
        x1 = layers.Reshape((128, 1))(x1)

        x1 = layers.Masking()(x1)

        x2 = layers.Embedding(input_dim=5, output_dim=127, mask_zero=True, name='embedding_sequence_layer')(sequences_inputs)
        x3 = layers.Embedding(input_dim=3, output_dim=128, mask_zero=True, name='embedding_secshape_layer')(secshape_inputs)

        x = layers.Dot(axes=(2, 1))([x3, x1])
        x = layers.Concatenate(axis=-1)([x2, x])

        x = layers.Bidirectional(layers.LSTM(units=self.lstm_units, return_sequences=return_seq),
                                 merge_mode='sum', name='biLstm')(x)

        x = Dense_Layer(48, 64)(x)

        outputs = layers.Dense(self.NUM_CLASSES, activation="softmax", name='outputs')(x)

        model = keras.Model(inputs=[sequences_inputs, secshape_inputs, coords_3d_inputs], outputs=outputs, name=self.modelname)

        model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                      metrics=[tf.keras.metrics.sparse_categorical_accuracy])

        self.myModel = model

    def model_fit(self, train_dataset, val_dataset):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            verbose=1,
            restore_best_weights=True
        )

        self.myModel.fit(train_dataset, epochs=self.EPOCHS, validation_data=val_dataset,
                         callbacks=[early_stopping])

    def model_save(self, saveprefix='./model'):
        filename = f'{saveprefix}/{self.modelname}/{self.modelname}'
        self.myModel.save_weights(filename)
        self.myModel.save(f'{filename}.model')

    def load(self, model_path):
        self.myModel.load_weights(model_path)

    def model_summary(self):
        self.myModel.summary()

    def predict(self, x):
        y_pre = self.myModel.predict(x)
        return y_pre

    def evaluate(self, dataset):
        loss, acc = self.myModel.evaluate(dataset)
        return loss, acc

def train_model(output_dataset_path, data_name, save_prefix, EPOCHS, base_lr, lstm_units):
    num_points = 1426
    BATCH_SIZE = 64

    model = NeuroPRIS(num_points=num_points, EPOCHS=EPOCHS, dataname=data_name,
                      learning_rate=base_lr, lstm_units=lstm_units)
    model.model_build()
    model.model_summary()

    train_dataset = tf.data.Dataset.load(f"{output_dataset_path}/train_dataset")
    val_dataset = tf.data.Dataset.load(f"{output_dataset_path}/val_dataset")

    def preprocess_sample(inputs, outputs, seq_ids):
        return inputs, outputs

    train_dataset = train_dataset.map(preprocess_sample)
    val_dataset = val_dataset.map(preprocess_sample)

    train_dataset = train_dataset.shuffle(800).batch(batch_size=BATCH_SIZE)
    val_dataset = val_dataset.shuffle(800).batch(batch_size=BATCH_SIZE)

    model.model_fit(train_dataset, val_dataset)

    loss, acc = model.evaluate(val_dataset)
    print(f'Evaluated Loss: {loss}, Evaluated Accuracy: {acc}')

    model.model_save(saveprefix=save_prefix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RNA model.')
    parser.add_argument('--output_dataset_path', type=str, required=True, help='Path where datasets were saved.')
    parser.add_argument('--data_name', type=str, required=True, help='Name of the dataset to be trained.')
    parser.add_argument('--save_prefix', type=str, default='./model', help='Prefix for model save directories.')
    parser.add_argument('--EPOCHS', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--base_lr', type=float, default=0.001, help='Base learning rate for training.')
    parser.add_argument('--lstm_units', type=int, default=48, help='Number of LSTM units.')
    args = parser.parse_args()

    train_model(args.output_dataset_path, args.data_name, args.save_prefix, args.EPOCHS, args.base_lr, args.lstm_units)