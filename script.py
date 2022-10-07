# Version 2

# Compatible with GMC, and CC datasets
# Using range instead of std in denom for mean normalization to avoid saving all X values to calculate within production.

# Computing mean normalization of entire data rather then on splits to capture the affect of entire distribution.
# Can save the mean and N for feature drift and normalization on test set.

import os
import math
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

import importlib
# import GAN_171103
# import xgboost as xgb0
# from GAN_171103 import *
# importlib.reload(GAN_171103)
from sklearn.ensemble import AdaBoostClassifier


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Reshape

from imblearn import datasets
from gsmote import GeometricSMOTE
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, KMeansSMOTE
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

np.random.seed(0)

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
		maxLabel = df[label].value_counts().idxmax()

		if minLabel == 0:
			df[label] = df[label].map({0 : 1, 1 : 0})
		elif maxLabel == 2:
			df[label] = df[label].map({1 : 1, 2: 0})

		indepCols = [col for col in df.columns if col != label]
		denom = (df[indepCols].max() - df[indepCols].min())
		df[indepCols] = (df[indepCols] - df[indepCols].mean()) / denom
			
		train, test = train_test_split(df, test_size=0.2, random_state=0,
			stratify = df[label])

		train.to_csv('%s_train.csv'%outFname, index = False)
		test.to_csv('%s_test.csv'%outFname, index = False)
		del test

	return outFname, train.copy()

def trainGANS(train, label_cols, X_col, MODEL_CACHE_RELATIVE_DIR):

	epochs = 5001

	MODEL_CACHE_DIR = '%s/cache/%s/'%(os.getcwd(),
		MODEL_CACHE_RELATIVE_DIR.split('/')[0])

	modelPath = '%s%s_generator_model_weights_step_%d.h5'%(
		MODEL_CACHE_DIR, MODEL_CACHE_RELATIVE_DIR.split(
			'/')[1], (epochs - 1))

	if not os.path.exists(modelPath):

		k_d = 1
		k_g = 1

		if not os.path.isdir(MODEL_CACHE_DIR):
			os.makedirs(MODEL_CACHE_DIR)
		
		train_no_label = train[X_col]

		arguments = [32, epochs, 64,  k_d, k_g, 100,
			1000, 0.0001, 128, MODEL_CACHE_DIR,
			None, None, None, False, MODEL_CACHE_RELATIVE_DIR]

		adversarial_training_WGAN(arguments, train, data_cols=X_col,
			label_cols=label_cols)

	return modelPath

def findSynthesizedData(data_cols, label_cols, train, modelPath, gen_samples):

	rand_dim = 32
	data_dim = len(data_cols)
	label_dim = len(label_cols)

	generator_model, discriminator_model, combined_model = define_models_CGAN(
		rand_dim, data_dim, label_dim, 128)
	
	generator_model.load_weights(modelPath)
	z = np.random.normal(size=(gen_samples, rand_dim))
	labels = np.array([[1.]] * gen_samples)
	g_z = generator_model.predict([z, labels])
	dfTrueGen = pd.DataFrame(g_z, columns = (data_cols + label_cols))
	trainBalanced = train.append(dfTrueGen)
	return trainBalanced.copy()

def sigmoid(z):
	
	s = 1/(1+np.exp(-z))
	cache = z
	return s,cache

def relu(z):
	
	r = np.maximum(0,z)
	cache = z
	return r,cache

def relu_backward(dA, cache):
	
	Z = cache
	dZ = np.array(dA, copy=True) # just converting dz to a correct object.
	
	# When z <= 0, you should set dz to 0 as well. 
	dZ[Z <= 0] = 0
	
	assert (dZ.shape == Z.shape)
	
	return dZ

def sigmoid_backward(dA, cache):

	Z = cache
	
	s = 1/(1+np.exp(-Z))
	dZ = dA * s * (1-s)
	
	assert (dZ.shape == Z.shape)
	
	return dZ

def linear_forward(A, W, b):

	Z = np.dot(W,A)+b
	cache = (A, W, b)
	
	return Z, cache

def linear_activation_forward(A_prev, W, b, activation):

	if activation == "sigmoid":
		Z, linear_cache = linear_forward(A_prev,W,b)
		A, activation_cache = sigmoid(Z)

	
	elif activation == "relu":
		Z, linear_cache = linear_forward(A_prev,W,b)
		A, activation_cache = relu(Z)
	
	cache = (linear_cache, activation_cache)

	return A, cache

def forward_propagation(X, parameters):
	caches = []
	A = X
	L = len(parameters) // 2                  # number of layers in the neural network
	
	# Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
	for l in range(1, L):
		A, cache = linear_activation_forward(A,parameters["W" + str(l)],parameters["b" + str(l)],activation="relu")
		caches.append(cache)
	
	# Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
	AL, cache = linear_activation_forward(A,parameters["W" + str(L)],parameters["b" + str(L)],activation="sigmoid")
	caches.append(cache)
			
	return AL, caches

def cost_function(AL, Y):
	m = Y.shape[1]

	cost = (-1/m)*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))

	cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
	
	return cost

def linear_backward(dZ, cache):
	A_prev, W, b = cache
	m = A_prev.shape[1]

	dW = (1/m)*np.dot(dZ,A_prev.T)
	db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
	dA_prev = np.dot(W.T,dZ)
	
	return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):

	linear_cache, activation_cache = cache
	
	if activation == "relu":
		dZ = relu_backward(dA,activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)
		
	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA,activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)
	
	return dA_prev, dW, db

class VAE(tf.keras.Model):

	def __init__(self, ndim, latent_dim):
		super(VAE, self).__init__()
		self.latent_dim = latent_dim  
		self.ndim = ndim        
		self.inference_net = Sequential(
			[
				InputLayer(input_shape=(ndim,)),
				# Dense(100, activation=tanh),
				Dense(2 * latent_dim)
			]
		)
		
		self.generative_net = Sequential(
			[
				InputLayer(input_shape=(latent_dim,)),
				# Dense(100, activation=tanh),
				Dense(ndim)
			])

	@tf.function
	def sample(self, num_samples=100, eps=None):
		if eps is None:
			eps = tf.random.normal(shape=(num_samples, self.latent_dim))
		return self.decode(eps)

	def encode(self, x):
		mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
		return mean, logvar

	def reparameterize(self, mean, logvar):
		eps = tf.random.normal(shape=mean.shape)
		return eps * tf.exp(logvar * 0.5) + mean

	def decode(self, z):
		return self.generative_net(z)

def log_normal_pdf(sample, mean, logvar, raxis=1):
	log2pi = tf.math.log(2. * np.pi)
	return tf.reduce_sum(-0.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
						 axis=raxis)

@tf.function
def compute_loss(model, x):
	mean, logvar = model.encode(x)
	logvar = tf.clip_by_value(logvar, -88., 88.)
	z = model.reparameterize(mean, logvar)
	xmean = model.decode(z)
	logpx_z = -tf.reduce_sum((x - xmean) ** 2, axis=1)
	logpz = log_normal_pdf(z, 0.0, 0.0)
	logqz_x = log_normal_pdf(z, mean, logvar)
	return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def compute_apply_gradients(model, x, optimizer):
	with tf.GradientTape() as tape:
		loss = compute_loss(model, x)
	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def trainVAE(xtrain, folderName):

	latent_dim = 2
	num_train, ndim = xtrain.shape
	MODEL_DIR = '%s/cache/%s/'%(os.getcwd(), folderName)

	if not os.path.isdir(MODEL_DIR):
		os.makedirs(MODEL_DIR)

	try:
		model = VAE(ndim, latent_dim)		
		model.load_weights(filepath=MODEL_DIR)
	except:
		epochs = 2000
		batch_size = 32
		optimizer = tf.keras.optimizers.Adam(1e-3)

		train_dataset = tf.data.Dataset.from_tensor_slices(
			xtrain.values.astype(np.float32)).shuffle(num_train).batch(
			batch_size)

		for epoch in range(1, epochs + 1):
			for train_x in train_dataset:
				compute_apply_gradients(model, train_x, optimizer)

		model.save_weights(MODEL_DIR)
		
	return model

def augment_data_interpolation(data, model, num_samples):
	
	X = data[data['Class'] == 1].drop(['Class'], axis=1)
	z, _ = model.encode(X.values.astype(np.float32))
	z1 = pd.DataFrame(z).sample(frac=num_samples / len(z), replace=True)
	z2 = z1.sample(frac=1)
	r = np.random.rand(*z1.shape)
	z = r * z1.values + (1 - r) * z2.values
	samples = model.decode(z.astype(np.float32)).numpy()
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


# def main(folderName = 'GMC', label = 'Class', synOffset = 5000, rand_dim = 32,
# 	base_n_count = 128, epochs = 50001, batch_size = 64,
# 	learning_rate = 0.0001, d_pre_train_steps = 100,
# 	syn_model = 'smote', skip =False):

def main(folderName = 'GMC', label = 'Class', synOffset = 5000,
	syn_model = 'smote', isIHT = False):

	# SMOTE done
	# GSMote done
	# KMSmote done

	# Gaussian Bernouli RBM
	# GAN
	# VAE	done
	# VAE + IHT	done

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

	if syn_model in ['smote', 'gsmote', 'ksmote']:
		for chunkToAdd in synChunks:
			print(chunkToAdd)

			if chunkToAdd == 0:
				trainFinal = train.copy()
			else:
				NMin = min(repList) + chunkToAdd
				NMax = max(repList) - chunkToAdd
				iht = InstanceHardnessThreshold(random_state=0, sampling_strategy = min(repList) / NMax)

				sampling_strategy =  NMin / NMax if isIHT else NMin / max(repList)

				if syn_model == 'smote':
					sm = SMOTE(random_state=0, sampling_strategy = sampling_strategy)

				elif syn_model == 'ksmote':

					sm = KMeansSMOTE(random_state=0, sampling_strategy = sampling_strategy,
						kmeans_estimator = 100)

				elif syn_model == 'gsmote':
					sm = GeometricSMOTE(random_state=0, sampling_strategy = sampling_strategy)

				X,y = train[X_col].values, train[label].values

				if isIHT:
					X, y = iht.fit_resample(X, y)

				X_res, y_res = sm.fit_resample(X, y)

				trainFinal = pd.DataFrame(data = X_res, columns = X_col)
				trainFinal[label] = y_res

			performObj['Synthesize Method'].append(syn_name)
			performObj['Synthesize Chunk'].append(chunkToAdd)
			performObj = applyModel(trainFinal, outFname, X_col, label, performObj)

	elif syn_model == 'vae':
		xtrain = train[train[label] == 1][X_col]
		vaem = trainVAE(xtrain, folderName)

		for chunkToAdd in synChunks:
			print(chunkToAdd)
			if chunkToAdd == 0:
				trainFinal = train.copy()
			else:
				trainFinal = augment_data_interpolation(
					train, vaem, chunkToAdd)

			performObj['Synthesize Method'].append(syn_name)
			performObj['Synthesize Chunk'].append(chunkToAdd)
			performObj = applyModel(trainFinal, outFname, X_col, label, performObj)

	elif syn_model == 'gan':

		xtrain = train[train[label] == 1][X_col]
		MODEL_CACHE_RELATIVE_DIR = "%s/gan/"%folderName
		modelPath = trainGANS(train, [label], X_col, MODEL_CACHE_RELATIVE_DIR)

		for chunkToAdd in synChunks:
			print(chunkToAdd)
			if chunkToAdd == 0:
				trainFinal = train.copy()
			else:
				trainFinal = findSynthesizedData(X_col, [label],
					train.copy(), modelPath)

			performObj = applyModel(trainFinal, outFname, X_col, label, performObj)

	dfPerform = pd.DataFrame(performObj)
	return dfPerform

def iterateModels():

	syn_list = []
	dataset = 'CC'
	writer = pd.ExcelWriter('%s_performance.xlsx'%dataset, engine='xlsxwriter')

	metrics = ['TP', 'TN', 'FP', 'FN', 'Precision', 'Sensitivity',
		'Specificity', 'AUC', 'Accuracy', 'G Mean', 'Balanced Accuracy',
		'F1 Score']

	for syn_method in ['smote', 'gsmote', 'ksmote']:
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