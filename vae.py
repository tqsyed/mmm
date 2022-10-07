# Version 2

# Compatible with GMC, and CC datasets
# Using range instead of std in denom for mean normalization to avoid saving all X values to calculate within production.

# Computing mean normalization of entire data rather then on splits to capture the affect of entire distribution.
# Can save the mean and N for feature drift and normalization on test set.

import os
import math
import glob
import random
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

import keras
import tensorflow as tf
from keras import layers
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# import xgboost as xgb0
from sklearn.ensemble import AdaBoostClassifier

from imblearn import datasets
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


# Set seeds to make the experiment more reproducible.
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 0
seed_everything(seed)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)



def dataIngestPreProp(folderName, label):

    DATA_DIR = '%s/data/%s/'%(os.getcwd(), folderName)
    PROCESSED_DATA_DIR = '%sprocessed/train_test/'%DATA_DIR

    if not os.path.isdir(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    fname = glob.glob('%s/*.csv'%DATA_DIR)[0]
    outFname = '%s%s'%(PROCESSED_DATA_DIR, fname.split('/')[-1].split('.')[0])

    if os.path.isfile('%s_train.csv'%outFname):

        print('Ingesting preprocessed data within memory')
        train = pd.read_csv('%s_train.csv'%outFname)

    else:
        print('Ingesting data within memory, and performing pre-processing')
        df = pd.read_csv(fname)
        minLabel = df[label].value_counts().idxmin()

        if minLabel == 0:
            df[label] = df[label].map({0: 1, 1: 0})

        indepCols = [col for col in df.columns if col != label]
        denom = (df[indepCols].max() - df[indepCols].min())
        df[indepCols] = (df[indepCols] - df[indepCols].mean()) / denom
            
        train, test = train_test_split(df, test_size=0.2, random_state=0,
            stratify = df[label])

        train.to_csv('%s_train.csv'%outFname, index = False)
        test.to_csv('%s_test.csv'%outFname, index = False)
        del test

    return outFname, train.copy()

def sampling(args):
    z_mean, z_log_sigma, latent_dim = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.)

    return z_mean + K.exp(z_log_sigma) * epsilon

def trainVAE(xtrain):

    x_train, x_test = train_test_split(xtrain, test_size=0.3, random_state=0)

    original_dim = x_train.shape[1]
    latent_dim = 2

    #GMC
    # batch_size = 32
    # epochs = 10

    #PH and SS
    # batch_size = 32
    # epochs = 50

    #CC
    batch_size = 64
    epochs = 50

    inputs = keras.Input(shape=(original_dim,))
    
    #GMC
    # h = layers.Dense(5, activation='relu')(inputs)

    # CC
    h = layers.Dense(30, activation='relu')(inputs)
    h = layers.Dense(28, activation='relu')(h)
    h = layers.Dense(26, activation='relu')(h)
    h = layers.Dense(26, activation='relu')(h)
    h = layers.Dense(16, activation='relu')(h)
    h = layers.Dense(8, activation='relu')(h)

    #PH
    # h = layers.Dense(68, activation='relu')(inputs)
    # h = layers.Dense(30, activation='relu')(h)


    # SS
    # h = layers.Dense(4, activation='relu')(inputs)
    # h = layers.Dense(3, activation='relu')(h)

    z_mean = layers.Dense(latent_dim)(h)
    z_log_sigma = layers.Dense(latent_dim)(h)

    z = layers.Lambda(sampling)([z_mean, z_log_sigma, latent_dim])

    # Create encoder
    encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

    # Create decoder
    latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
    
    #GMC
    # x = layers.Dense(5, activation='relu')(latent_inputs)
    
    #CC
    x = layers.Dense(8, activation='relu')(latent_inputs)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(24, activation='relu')(x)
    x = layers.Dense(26, activation='relu')(x)
    x = layers.Dense(28, activation='relu')(x)
    x = layers.Dense(30, activation='relu')(x)

    #SS
    # x = layers.Dense(3, activation='relu')(latent_inputs)
    # x = layers.Dense(4, activation='relu')(x)

    #PH
    # x = layers.Dense(30, activation='relu')(latent_inputs)
    # x = layers.Dense(68, activation='relu')(x)

    # outputs = layers.Dense(original_dim, activation='sigmoid')(x)
    outputs = layers.Dense(original_dim, activation=None)(x)
    
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = keras.Model(inputs, outputs, name='vae_mlp')

    # reconstruction_loss = mse(x, decoded_mean)
    reconstruction_loss = K.sum(K.square(inputs - outputs), axis=-1)
    kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    early_stopping = EarlyStopping(monitor='loss', patience=10, min_delta=1e-5) #stop training if loss does not decrease with at least 0.00001
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, min_delta=1e-5, factor=0.2) #reduce learning rate (divide it by 5 = multiply it by 0.2) if loss does not decrease with at least 0.00001
    callbacks = [early_stopping, reduce_lr]

    vae.fit(x_train, x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, x_test),
            callbacks=callbacks)

    return decoder, latent_dim

def augment_data_interpolation(data, decoder, num_samples, latent_dim):
    
    r = np.random.rand(*[num_samples,latent_dim])
    samples = decoder.predict(r)
    dfnew = pd.DataFrame(samples, columns=data.columns.drop('Class'))
    dfnew['Class'] = np.ones(len(samples), dtype=np.int)
    dfnew = pd.concat((data, dfnew), ignore_index=True).sample(frac=1)

    return dfnew

def applyModel(train_df, outFname, X_col, y_col, performObj):

    test_df = pd.read_csv('%s_test.csv'%outFname)
    X_test, y_test = test_df[X_col], test_df[y_col]
    X_train, y_train = train_df[X_col], train_df[y_col]

    # clf = xgb.XGBClassifier(random_state = 0)
    clf = AdaBoostClassifier(random_state = 0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_true = test_df[y_col].values

    return modelExecutionMetrics(y_true, y_pred, performObj)

def modelExecutionMetrics(y_true, y_pred, performObj):

    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = TP / (TP + FP)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    performObj['TP'].append(TP)
    performObj['TN'].append(TN)
    performObj['FP'].append(FP)
    performObj['FN'].append(FN)

    performObj['Precision'].append(precision)
    performObj['Sensitivity'].append(sensitivity)
    performObj['Specificity'].append(specificity)
    performObj['AUC'].append(roc_auc_score(y_true,y_pred))
    performObj['Accuracy'].append((TP + TN) / (TP + TN + FP + FN))
    performObj['G Mean'].append(np.sqrt(sensitivity * specificity))
    performObj['Balanced Accuracy'].append(balanced_accuracy_score(y_true, y_pred))
    performObj['F1 Score'].append(2 * ((precision * sensitivity) / (precision + sensitivity)))

    return performObj

def incorporateProteinDataset():

    data = datasets.fetch_datasets()['protein_homo']
    df = pd.DataFrame(data.data)
    df['Class'] = data.target
    df['Class'] = df['Class'].replace(-1,0)

def main(folderName = 'GMC', label = 'Class', synOffset = 5000,
    syn_model = 'vae', isIHT = False):

    performObj = defaultdict(list)
    outFname, train = dataIngestPreProp(folderName, label)
    X_col = train.columns.tolist()
    X_col.remove(label)

    X,y = train[X_col].values, train[label].values
    syn_name = syn_model.upper()
    repList = train[label].value_counts().values
    minorityDeficit = abs(repList[0] - repList[1])

    synChunks = list(range(0, minorityDeficit,
        synOffset)) + [minorityDeficit]

    if isIHT:
        syn_name += ' + IHT'
        limit = math.floor((max(repList) - min(repList))/2/synOffset) * synOffset
        synChunks = [chunk for chunk in synChunks if chunk <= limit]

    print(syn_name)

    if syn_model == 'vae':

        xtrain = train[train[label] == 1][X_col]
        decoder, latent_dim = trainVAE(xtrain.values)

        for chunkToAdd in synChunks:
            print(chunkToAdd)

            if chunkToAdd == 0:
                trainFinal = train.copy()
            else:

                NMin = min(repList) + chunkToAdd
                NMax = max(repList) - chunkToAdd
                iht = InstanceHardnessThreshold(random_state=0,
                    sampling_strategy = min(repList) / NMax)

                if isIHT:
                    XI, yI = iht.fit_resample(X, y)
                    train = pd.DataFrame(XI, columns = X_col)
                    train[label] = yI

                trainFinal = augment_data_interpolation(
                    train, decoder, chunkToAdd, latent_dim)

            performObj['Synthesize Method'].append(syn_name)
            performObj['Synthesize Chunk'].append(chunkToAdd)
            performObj = applyModel(trainFinal, outFname, X_col, label, performObj)

    dfPerform = pd.DataFrame(performObj)
    return dfPerform

def iterateModels():

    syn_list = []
    dataset = 'CC'

    writer = pd.ExcelWriter('%s_vae_performance.xlsx'%dataset, engine='xlsxwriter')

    metrics = ['TP', 'TN', 'FP', 'FN', 'Precision', 'Sensitivity',
        'Specificity', 'AUC', 'Accuracy', 'G Mean', 'Balanced Accuracy',
        'F1 Score']

    # for syn_method in ['smote', 'gsmote', 'ksmote', 'vae']:
    for syn_method in ['vae']:
        for isIHT in [True, False]:
            dfPerform = main(syn_model = syn_method, folderName = dataset, isIHT = isIHT)
            syn_list.append(dfPerform)

    df = pd.concat(syn_list)

    for metric in metrics:
        dfMetric = df.pivot(
            index = 'Synthesize Chunk',
            columns = 'Synthesize Method',
            values = metric).add_prefix('%s_'%metric)

        dfMetric.to_excel(writer, sheet_name=metric)
    writer.save()