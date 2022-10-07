# Version 2

# Compatible with GMC, and CC datasets
# Using range instead of std in denom for mean normalization to avoid saving all X values to calculate within production.

# Computing mean normalization of entire data rather then on splits to capture the affect of entire distribution.
# Can save the mean and N for feature drift and normalization on test set.

# from __future__ import print_function, division

import os
import math
import glob
import random
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

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

def trainGANS(train, label, rand_dim, base_n_count, epochs,
    batch_size, learning_rate, d_pre_train_steps, model,
    folderName):

    print('Training GANS')

    label_cols = [label]
    data_cols = [col for col in train.columns if col != label_cols[0]]

    k_d = 1  # number of critic network updates per adversarial training step
    k_g = 1  # number of generator network updates per adversarial training step
    
    train_no_label = train[train[label] == 1][data_cols]

    arguments = [rand_dim, epochs, batch_size,  k_d, k_g, d_pre_train_steps,
        1000, learning_rate, base_n_count, MODEL_CACHE_DIR, None, None, None, True, folderName]

    if model == 'gan':
        adversarial_training_GAN(arguments, train_no_label, data_cols)
        
    elif model == 'cgan':
        adversarial_training_GAN(arguments, train, data_cols=data_cols,
            label_cols=label_cols)

    elif model == 'wgan':
        adversarial_training_WGAN(arguments, train_no_label, data_cols=data_cols)

    elif model == 'wcgan':
        adversarial_training_WGAN(arguments, train, data_cols=data_cols,
            label_cols=label_cols)

    return data_cols, label_cols, modelPath

def main(folderName = 'GMC', label = 'Class', synOffset = 5000,
    syn_model = 'vae', isIHT = False):

    performObj = defaultdict(list)
    outFname, train = dataIngestPreProp(folderName, label)
    X_col = train.columns.tolist()
    X_col.remove(label)

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

    if syn_model == 'gan':

        rand_dim = 2
        base_n_count = 2
        epochs = 100
        batch_size = 32

        data_cols, label_cols, modelPath = trainGANS(train, label, rand_dim, base_n_count, epochs, batch_size,
            learning_rate, d_pre_train_steps, model, '%s/%s-%s-%s/'%(
                folderName, model, mntype, mndenom))

        # for chunkToAdd in synChunks:
        #     print(chunkToAdd)

        #     if chunkToAdd == 0:
        #         trainFinal = train.copy()
        #     else:
        #         trainFinal = ctgan.sample(chunkToAdd)
        #         trainFinal['Class'] = np.ones(trainFinal.shape[0], dtype=np.int)
        #         trainFinal = pd.concat((train, trainFinal), ignore_index=True).sample(frac=1)

        #     performObj['Synthesize Method'].append(syn_name)
        #     performObj['Synthesize Chunk'].append(chunkToAdd)
        #     performObj = applyModel(trainFinal, outFname, X_col, label, performObj)

    dfPerform = pd.DataFrame(performObj)
    return dfPerform

def iterateModels():

    syn_list = []
    dataset = 'EEG'

    writer = pd.ExcelWriter('%s_gan_performance.xlsx'%dataset, engine='xlsxwriter')

    metrics = ['TP', 'TN', 'FP', 'FN', 'Precision', 'Sensitivity',
        'Specificity', 'AUC', 'Accuracy', 'G Mean', 'Balanced Accuracy',
        'F1 Score']

    # for syn_method in ['smote', 'gsmote', 'ksmote', 'vae']:
    for syn_method in ['gan']:
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