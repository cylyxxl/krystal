from tkinter import X
from turtle import xcor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

df = pd.read_feather('/data/xxl//dataset/train.feather')
print(df.head(10))
print(df.columns)

f_col = df.drop(['row_id','time_id','investment_id','target'],axis=1).columns
print(f_col)

scaler = StandardScaler()
scaler.fit(pd.DataFrame(df['investment_id']))

def make_dataset(df):
    inv_df = df['investment_id']
    f_df = df[f_col]
    scaled_investment_id = scaler.transform(pd.DataFrame(inv_df))
    df['investment_id'] = scaled_investment_id
    data_x = pd.concat([df['investment_id'], f_df], axis=1)
    return data_x

df=df.astype('float16')
df_x = make_dataset(df)
print(df_x.head(10))

df_y = pd.DataFrame(df['target'])
print(df_y)


def pythonash_model():
    inputs_ = tf.keras.Input(shape=[df_x.shape[1]])
    x = tf.keras.layers.Dense(64, kernel_initializer='he_normal',activation='swish')(inputs_)
    # batch = tf.keras.layers.BatchNormalization()(x)
    # leaky = tf.keras.layers.LeakyReLU(0.1)(batch)

    x = tf.keras.layers.Dense(128, kernel_initializer='he_normal',activation='swish')(x)
    # batch = tf.keras.layers.BatchNormalization()(x)
    # leaky = tf.keras.layers.LeakyReLU(0.1)(batch)

    x = tf.keras.layers.Dense(256, kernel_initializer='he_normal',activation='swish')(x)
    # batch = tf.keras.layers.BatchNormalization()(x)
    # leaky = tf.keras.layers.LeakyReLU(0.1)(batch)

    x = tf.keras.layers.Dense(512, kernel_initializer='he_normal',activation='swish')(x)
    # batch = tf.keras.layers.BatchNormalization()(x)
    # leaky = tf.keras.layers.LeakyReLU(0.1)(batch)

    x = tf.keras.layers.Dense(256, kernel_initializer='he_normal',activation='swish')(x)
    batch = tf.keras.layers.BatchNormalization()(x)
    # leaky = tf.keras.layers.LeakyReLU(0.1)(batch)
    # drop = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(128, kernel_initializer='he_normal',activation='swish')(x)
    # batch = tf.keras.layers.BatchNormalization()(x)
    # leaky = tf.keras.layers.LeakyReLU(0.1)(batch)

    x = tf.keras.layers.Dense(8, kernel_initializer='he_normal',activation='swish')(x)
    # batch = tf.keras.layers.BatchNormalization()(x)
    # leaky = tf.keras.layers.LeakyReLU(0.1)(batch)
    # drop = tf.keras.layers.Dropout(0.4)(X)

    outputs_ = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs_, outputs=outputs_)

    rmse = tf.keras.metrics.RootMeanSquaredError()

    # learning_sch = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=0.003,
    #     decay_steps=9700,
    #     decay_rate=0.98)
    # adam = tf.keras.optimizers.Adam(learning_rate=learning_sch)

    model.compile(loss='mse', metrics=rmse, optimizer=tf.optimizers.Adam(0.001))
    return model


pythonash_model().summary()
from tensorflow.keras.utils import  plot_model

plot_model(pythonash_model(),to_file = '/workspace/xxl/modle.png',show_shapes=True,expand_nested=True)

kfold_generator = KFold(n_splits =5, shuffle=True, random_state = 2022)
print(kfold_generator)
callbacks = tf.keras.callbacks.ModelCheckpoint('pythonash_model.h5', save_best_only=True)
for train_index, val_index in kfold_generator.split(df_x, df_y):
    train_x, train_y = df_x.iloc[train_index], df_y.iloc[train_index]
    val_x, val_y = df_x.iloc[val_index], df_y.iloc[val_index]
    tf_train = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(2022).batch(1024,drop_remainder=False).prefetch(
        1)
    tf_val = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(1024,drop_remainder=False).prefetch(
        1)
    model = pythonash_model()
    model.fit(tf_train, callbacks=callbacks, epochs=5,  #### change the epochs into more numbers.
              validation_data=(tf_val))
    corr = pearsonr(model.predict(tf_val).ravel(),val_y.values.ravel())
    print(corr)


    # print('Pearson correlation score: {corr}')  