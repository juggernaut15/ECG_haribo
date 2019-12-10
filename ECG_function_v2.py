#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import neurokit as nk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os, shutil
import numpy as np
from os import listdir

import scipy
from scipy import signal
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
import os, shutil
from scipy import sparse
from scipy.sparse.linalg import spsolve
import peakutils
from biosppy import storage
from biosppy.signals import ecg
import random
from keras import layers
from keras import models
from keras import optimizers

get_ipython().run_line_magic('matplotlib', 'inline')
import keras
from sklearn import datasets
#from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras import backend as K
from sklearn.model_selection import train_test_split
import scipy.io.wavfile as wav
import scipy.signal as signal
from matplotlib import pyplot as plt
import pywt
import sys


# In[ ]:


from keras.models import load_model
from keras.models import Model
from numpy.fft import fft, ifft, fft2, ifft2, fftshift
import itertools
from numpy import dot
from numpy.linalg import norm


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBClassifier, plot_importance

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score



num_file = 4 # 등록 + 인증 파일의 개수
num_peak_per_file = 15 # 20초 측정 파일에서 사용할 peak 개수


class ECG_options:
    def __init__(self, Model_name, Num_people, Len_resample, Num_resample_per_data, Type_resample, Ft_type, Num_train_people):
        self.model_name = Model_name
        self.num_people = Num_people
        self.len_resample = Len_resample
        self.num_resample_per_data = Num_resample_per_data
        self.type_resample = Type_resample
        self.ft_type = Ft_type
        self.num_train_people = Num_train_people
        if Type_resample == 'peak':
            self.num_resample_per_file = num_peak_per_file
        elif Type_resample == 'cycle':
            self.num_resample_per_file = num_peak_per_file - 1
    def print_options(self, f):
        print('model : ' + self.model_name + '  num_people : ' + str(self.num_people), file = f)
        print('len_resample : ' + str(self.len_resample) + '  num_resample_per_data : ' + str(self.num_resample_per_data), file = f)
        print('type_resample : ' + self.type_resample + '  ft_type : ' + self.ft_type, file = f)
        
        
# In[ ]:


def cal_threshold(a, b, type_thr):
    result = 0
    if type_thr == 'L1':
        result = 1/(1 + norm(a - b))
    elif type_thr == 'prd':
        result = (norm(a - b)/norm(a))*100
    elif type_thr == 'corr':
        result = np.corrcoef(a,b, rowvar=False)
        result = result[0][1]
    elif type_thr == 'correlate':
        result = np.correlate(a, b, 'full')
    elif type_thr == 'cosine':
        result = dot(a, b)/(norm(a)*norm(b))
    else:
        print('type_thr : 잘못된 입력입니다.')
        return
    return result

 
def cross_correlation_using_fft(x, y):
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)
 
# shift &lt; 0 means that y starts 'shift' time steps before x # shift &gt; 0 means that y starts 'shift' time steps after x
def compute_shift(x, y):
    assert len(x) == len(y)
    c = cross_correlation_using_fft(x, y)
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift


# In[ ]:


def load_data(dataset_path, num_people):
    name_file = ['_{}.txt'.format(i+1) for i in range(num_file)]
    ecgdata = []
    for name in range(num_people):
        path = os.path.join(dataset_path, str(name))
        for fname in name_file:
            data_path = path + str(fname)
            data = pd.read_csv(data_path, delimiter = "\t")
            data = data.drop(data.columns[[0, 2, 3]], axis='columns')
            data.columns = ["A"]
            ecgdata.append(data)
    return ecgdata

def preprocessing_ecg(ecg_options, ecgdata):
    measures = []
    resamples = []
    
    num_data = ecg_options.num_people*num_file
    #len_data = (ecg_options.num_resample_per_data) * (ecg_options.len_resample)
    
    for i in range(num_data):
        filtering_detecting_peaks(ecgdata[i], measures)
        resampling_ecg(ecgdata[i], measures[i], ecg_options, resamples)     
    return resamples



def make_dataset(ecg_options, resamples):
    num_data_per_file = ecg_options.num_resample_per_file - ecg_options.num_resample_per_data + 1
    train_people_list = random.sample(range(ecg_options.num_people), ecg_options.num_train_people)
    len_data = (ecg_options.num_resample_per_data) * (ecg_options.len_resample)
    len_resample = ecg_options.len_resample
    
    
    train_dataset = []
    train_target = []
    train_index = 0
    
    for i in train_people_list:
        for j in range(num_file):
            for k in range(num_data_per_file):
                if ecg_options.ft_type == 'stft':
                     f, t, data = signal.stft(resamples[4*i+j][(len_resample)*k : (len_resample)*k + len_data], 1000)
                elif ecg_options.ft_type == 'dct':
                    data = scipy.fftpack.dct(resamples[4*i+j][(len_resample)*k : (len_resample)*k + len_data])
                elif ecg_options.ft_type == 'wavelet_1':
                    (data1, data2) =  pywt.dwt(resamples[4*i+j][(len_resample)*k : (len_resample)*k + len_data], 'db1')
                    data = np.concatenate((data1, data2), axis=0)
                elif ecg_options.ft_type == 'wavelet_2':
                    (data1, data2) =  pywt.dwt(resamples[4*i+j][(len_resample)*k : (len_resample)*k + len_data], 'db1')
                    data = data2
                elif ecg_options.ft_type == 'none':
                    data = resamples[4*i+j][(len_resample)*k : (len_resample)*k + len_data]
                else:
                    print('ft_type : 잘못된 입력입니다.')
                    return
                train_dataset.append(data)
                train_target.append(train_index)
        train_index = train_index + 1
    train_dataset = np.array(train_dataset)
    
    return train_dataset, train_target
        

def filtering_detecting_peaks(data, measures):
    measure = {}
    #templates의 길이는 sampling rate의 60%
    out = ecg.ecg(signal=data.A, sampling_rate=1000, show = False )
    data['filtered'] = out['filtered']
    measure['peaklist'] = out['rpeaks']
    measure['ybeat'] = [out['filtered'][x] for x in out['rpeaks']]
    measures.append(measure)
    return


def resampling_ecg(data, measure, ecg_options, resamples):
    resample_data = []
    resample_length = ecg_options.len_resample
    num_resample_per_file = ecg_options.num_resample_per_file
    
    if ecg_options.type_resample == 'cycle':
        for i in range(num_resample_per_file):
            b = signal.resample(data.filtered[measure['peaklist'][i+1]:measure['peaklist'][i+2]], resample_length)
            resample_data = np.concatenate((resample_data, b), axis=0)
        resamples.append(resample_data)
    elif ecg_options.type_resample == 'peak':
        for i in range(num_resample_per_file):
            b = data.filtered[measure['peaklist'][i+1] - (int)(resample_length/2) : measure['peaklist'][i+1] + (int)(resample_length/2)]
            resample_data = np.concatenate((resample_data, b), axis=0)
        resamples.append(resample_data)
    return

'''
def re_sampling_cycle(data, measure, resample_length, resamples):
    resample_data = []
    num_resample = len(measure['peaklist'])-1
    for i in range(num_resample):
        b = signal.resample(data.filtered[measure['peaklist'][i]:measure['peaklist'][i+1]], resample_length)
        resample_data = np.concatenate((resample_data, b), axis=0)
    resamples.append(resample_data) 
    return
    

# peak점 중심으로 일정 간격
def re_sampling_peak(data, measure, resample_length, resamples):
    resample_data = []
    num_resample = len(measure['peaklist'])
    for i in range(num_resample):
        b = data.filtered[measure['peaklist'][i] - (int)(resample_length/2) : measure['peaklist'][i] + (int)(resample_length/2)]
        resample_data = np.concatenate((resample_data, b), axis=0)
    resamples.append(resample_data)
    return

'''
# In[ ]:


def preparing_ecg(ecg_options):
    dataset_path = './data/' 
    num_data = ecg_options.num_people*num_file
    #num_peak_per_resamples = 15
    
    ecgdata = load_data(dataset_path, ecg_options.num_people)
    resamples = preprocessing_ecg(ecg_options, ecgdata)
    train_dataset, train_target = make_dataset(ecg_options, resamples)

    return  train_dataset, train_target
    

# In[ ]:f


def train_CNN_Conv1D(ecg_options):

    train_dataset, train_target = preparing_ecg(ecg_options)
    num_class = ecg_options.num_train_people
    len_data = len(train_dataset[0])
    
    X_train, X_test, y_train, y_test = train_test_split(train_dataset, train_target, stratify = train_target, test_size=0.25, random_state=1234)
    y_train_binary = keras.utils.to_categorical(y_train)
    y_test_binary = keras.utils.to_categorical(y_test)
    x_train = X_train.reshape(len(X_train), len_data, 1)
    x_test = X_test.reshape(len(X_test), len_data, 1)

    model_m = Sequential()
    model_m.add(Conv1D(filters = 16, kernel_size = 7,activation='relu',input_shape=(len_data, 1), strides=1))
    model_m.add(MaxPooling1D(3,strides=2))

    model_m.add(Conv1D(32, 5, activation='relu' ,strides=1))
    model_m.add(BatchNormalization())
    model_m.add(MaxPooling1D(3,strides=2))

    model_m.add(Conv1D(64, 5, activation='relu',strides=1))
    model_m.add(MaxPooling1D(3,strides=2))

    model_m.add(Conv1D(128, 7, activation='relu',strides=1))
    model_m.add(BatchNormalization())
    model_m.add(Conv1D(256, 7, activation='relu',strides=1))
    model_m.add(Conv1D(256, 8, activation='relu',strides=1))
    model_m.add(BatchNormalization())
    model_m.add(Dropout(0.5))
    model_m.add(Flatten())

    ###############################################################################################
    model_m.add(Dense(256, activation='relu'))
    model_m.add(Dense(num_class, activation = 'softmax'))

    learning_rate = 0.001
    decay = 0.0005
    sgd = optimizers.SGD(lr=learning_rate, decay=decay)

    model_m.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    batch = 50
    epochs = 80 
    history = model_m.fit(x_train, y_train_binary,
              batch_size = batch,
              epochs = epochs,
              verbose = 1,
              validation_split = 0.2)
    
    rr = model_m.evaluate(x_test, y_test_binary)
    a = ecg_options.len_resample
    b = ecg_options.num_resample_per_data
    c = ecg_options.type_resample
    d = ecg_options.ft_type
    e = ecg_options.num_train_people
    f = ecg_options.num_people
    m_name = f'ECG_CNN_Conv1D_{a}len_{b}{c}_{d}_ft_train{e}_{f}'
    save_path = './model'
    save_name = os.path.join(save_path, m_name)
    model_m.save(save_name)
    f = open('./output/CNN_train_output.txt','a')
    print(save_name + '\t', file = f)
    print(rr, file = f)
    print('\n', file = f)
    f.close()
    


# In[ ]:

def make_sum_enrollment_templates(ecg_options):
    train_dataset, train_target = preparing_ecg(ecg_options)
    len_data = len(train_dataset[0])
    num_data_per_file = ecg_options.num_resample_per_file - ecg_options.num_resample_per_data + 1

    if ecg_options.model_name == 'none':
        Train_dataset_template = train_dataset
    else:
        model_ = os.path.join('./model', ecg_options.model_name)
        Train_dataset = train_dataset.reshape(len(train_dataset), len_data, 1)
        model_m = load_model(model_)
        intermediate_layer_model = Model(inputs=model_m.input, outputs=model_m.layers[13].output)
        Train_dataset_template = intermediate_layer_model.predict(Train_dataset)
        
    len_template = len(Train_dataset_template[0])
    sum_enrollment_templates = []

    for i in range(num_file):
        templates = np.zeros((ecg_options.num_people, len_template))
        for j in range(ecg_options.num_people):
            for k in range(num_data_per_file):
                templates[j] += Train_dataset_template[i*num_data_per_file + j*train_target.count(0) + k]
        sum_enrollment_templates.append(templates)
    
    if ecg_options.model_name == 'none':
        templates_name = os.path.join('./sum_enrollment_templates', ecg_options.resample_type)
    else:
        templates_name = os.path.join('./sum_enrollment_templates', ecg_options.model_name)
    np.save(templates_name, sum_enrollment_templates)
    return

def make_sum_verification_templates(ecg_options):
    train_dataset, train_target = preparing_ecg(ecg_options)
    len_data = len(train_dataset[0])    
    num_data_per_file = ecg_options.num_resample_per_file - ecg_options.num_resample_per_data + 1

    if ecg_options.model_name == 'none':
        Train_dataset_template = train_dataset
    else:
        model_ = os.path.join('./model', ecg_options.model_name)
        Train_dataset = train_dataset.reshape(len(train_dataset), len_data, 1)
        model_m = load_model(model_)
        intermediate_layer_model = Model(inputs=model_m.input, outputs=model_m.layers[13].output)
        Train_dataset_template = intermediate_layer_model.predict(Train_dataset)
    
    #num_resample_per_file = (int)(train_target.count(0)/num_file)
    len_template = len(Train_dataset_template[0])
   
    for num_data_per_verification in range(1, num_data_per_file + 1):
        for l in range(num_data_per_file - num_data_per_verification + 1):
            sum_verification_templates = []
            for i in range(num_file):
                templates = np.zeros((ecg_options.num_people, len_template))
                for j in range(ecg_options.num_people):
                    for k in range(num_data_per_verification):
                        templates[j] += Train_dataset_template[l + i*num_data_per_verification + j*train_target.count(0) + k]
                sum_verification_templates.append(templates)
                
            if ecg_options.model_name == 'none':
                templates_name = os.path.join('./sum_verification_templates', str(num_data_per_verification), str(l) + ecg_options.resample_type)
            else:
                templates_name = os.path.join('./sum_verification_templates', str(num_data_per_verification), str(l) + ecg_options.model_name)
            np.save(templates_name, sum_verification_templates)
    return

def verification_test(ecg_options):
    num_file_verification = 1
    num_file_enrollment = 3
    num_data_per_file = ecg_options.num_resample_per_file - ecg_options.num_resample_per_data + 1
    #num_resample_per_file = (int)(train_target.count(0)/num_file)

    if ecg_options.model_name == 'none':
        return
    else:
        sum_enrollment_templates_name = os.path.join('./sum_enrollment_templates', ecg_options.model_name + '.npy')
        sum_enrollment_templates = np.load(sum_enrollment_templates_name)
        len_template = len(sum_enrollment_templates[0][0])

        for num_data_per_verification in range(1, num_data_per_file+1):
            for l in range(num_data_per_file - num_data_per_verification + 1):
                sum_verification_templates_name = os.path.join('./sum_verification_templates', 
                                                               str(num_data_per_verification), str(l) + ecg_options.model_name+'.npy')
                sum_verification_templates = np.load(sum_verification_templates_name)
                ####
                for i in range(num_file):
                    verification_templates = np.zeros((ecg_options.num_people, len_template))
                    enrollment_templates = np.zeros((ecg_options.num_people, len_template))
                    for j in range(num_file):
                        if (j !=  i):
                            enrollment_templates += sum_enrollment_templates[j]

                    verification_templates = sum_verification_templates[i]
                    verification_templates /= num_data_per_verification * num_file_verification
                    enrollment_templates /= num_data_per_file * num_file_enrollment

                    dist_list = ['cosine', 'L1', 'corr']
                    for dist in dist_list:
                        fnmr = []
                        fmr = []

                        for thr in range(100):
                            thr /= 100
                            fnm = 0
                            fm = 0
                            for j in itertools.combinations_with_replacement(range(ecg_options.num_people), 2):
                                val = cal_threshold(enrollment_templates[j[0]], verification_templates[j[1]], dist)
                                if val>thr:
                                    if j[0] == j[1]:
                                        fnm +=1
                                    else:
                                        fm += 1
                            fnmr.append(ecg_options.num_people-fnm)
                            fmr.append(fm)

                        FNMR = (np.array(fnmr)/ecg_options.num_people)*100
                        FMR = (np.array(fmr)/(ecg_options.num_people*(ecg_options.num_people-1))) * 50
                        res = abs(FNMR - FMR)
                        m = min(res)
                        for j in range(100):
                            if m == res[j]:
                                Sth = j

                        f = open('./output/verification_output.txt', 'a')    
                        ecg_options.print_options(f)
                        print('type_thr : ' + dist + '  num_data_per_verification : ' + str(num_data_per_verification), file = f)
                        print('index :', str(l), 'Sth :', str(Sth),' FNMR :',str(FNMR[Sth]), ' FMR :',str(FMR[Sth]), file = f)
                        print('\n', file = f)
                        f.close()
                
                
                
    '''
    model_ = os.path.join('./model', str(ecg_options.model_name))
    train_dataset, train_target = preparing_ecg(ecg_options)
    len_data = len(train_dataset[0])    

    if ecg_options.model_name == 'none':
        Train_dataset_template = train_dataset
    else:
        Train_dataset = train_dataset.reshape(len(train_dataset), len_data, 1)
        model_m = load_model(model_)
        intermediate_layer_model = Model(inputs=model_m.input, outputs=model_m.layers[13].output)
        Train_dataset_template = intermediate_layer_model.predict(Train_dataset)
        
    num_resample_per_file = (int)(train_target.count(0)/num_file)
    len_template = len(Train_dataset_template[0])
    num_file_verification = 1
    num_file_enrollment = 3
    
    for num_resample_per_verification in range(1, ecg_options.num_people+1):
        sum_enrollment_templates = []

        for i in range(num_file):
            templates = np.zeros((ecg_options.num_people, len_template))
            for j in range(ecg_options.num_people):
                for k in range(num_resample_per_file):
                    templates[j] += Train_dataset_template[i*num_resample_per_file + j*train_target.count(0) + k]
            sum_enrollment_templates.append(templates)

        for l in range(num_resample_per_file - num_resample_per_verification + 1):
            sum_verification_templates = []

            for i in range(num_file):
                templates = np.zeros((ecg_options.num_people, len_template))
                for j in range(ecg_options.num_people):
                    for k in range(num_resample_per_verification):
                        templates[j] += Train_dataset_template[l + i*num_resample_per_verification + j*train_target.count(0) + k]
                sum_verification_templates.append(templates)

            for i in range(num_file):
                verification_templates = np.zeros((ecg_options.num_people, len_template))
                enrollment_templates = np.zeros((ecg_options.num_people, len_template))
                for j in range(num_file):
                    if (j !=  i):
                        enrollment_templates += sum_enrollment_templates[j]

                verification_templates = sum_verification_templates[i]
                verification_templates /= num_resample_per_verification * num_file_verification
                enrollment_templates /= num_resample_per_file*num_file_enrollment

                dist_list = ['cosine', 'L1', 'corr']
                for dist in dist_list:
                    fnmr = []
                    fmr = []

                    for thr in range(0, 1000):
                        thr /= 1000
                        fnm = 0
                        fm = 0
                        for j in itertools.combinations_with_replacement(range(ecg_options.num_people), 2):
                            val = cal_threshold(enrollment_templates[j[0]], verification_templates[j[1]], dist)
                            if val>thr:
                                if j[0] == j[1]:
                                    fnm +=1
                                else:
                                    fm += 1
                        fnmr.append(ecg_options.num_people-fnm)
                        fmr.append(fm)

                    FNMR = (np.array(fnmr)/ecg_options.num_people)*100
                    FMR = (np.array(fmr)/(ecg_options.num_people*(ecg_options.num_people-1))) * 50
                    res = abs(FNMR - FMR)
                    m = min(res)
                    for j in range(1000):
                        if m == res[j]:
                            Sth = j

                    f = open('./output/verification_output.txt', 'a')    
                    ecg_options.print_options(f)
                    print('type_thr : ' + dist + '  num_resample_per_verification : ' + str(num_resample_per_verification), file = f)
                    print('index :', str(l), 'Sth :', str(Sth),' FNMR :',str(FNMR[Sth]), ' FMR :',str(FMR[Sth]), file = f)
                    print('\n', file = f)
                    f.close()

        
        
    
    for i in range(ecg_options.num_people):
        idx = i*train_target.count(0)
        for j in range(idx, idx + num_resample_per_file * num_file_enrollment):
            enrollment_template[i] += Train_dataset_template[j]
        for j in range(idx + num_resample_per_file * (num_file_enrollment), idx + num_resample_per_file * num_file):
            verification_template[i] += Train_dataset_template[j]

    for i in range(ecg_options.num_people):
        idx = i*train_target.count(0)
        for j in range(idx, idx + num_resample_per_file * num_file_enrollment):
            enrollment_template[i] += Train_dataset_template[j]
        for j in range(idx + num_resample_per_file * (num_file_enrollment), idx + num_resample_per_file * num_file):
            verification_template[i] += Train_dataset_template[j]

    verification_template /= num_resample_per_file*num_file_verification
    enrollment_template  /= num_resample_per_file*num_file_enrollment

    fnmr = []
    fmr = []

    for thr in range(0, 100):
        thr /= 100
        fnm = 0
        fm = 0
        for i in itertools.combinations_with_replacement(range(ecg_options.num_people), 2):
            val = cal_threshold(enrollment_template[i[0]], verification_template[i[1]], ecg_options.type_thr)
            if val>thr:
                if i[0] == i[1]:
                    fnm +=1
                else:
                    fm += 1
        fnmr.append(ecg_options.num_people-fnm)
        fmr.append(fm)

    FNMR = (np.array(fnmr)/ecg_options.num_people)*100
    FMR = (np.array(fmr)/(ecg_options.num_people*(ecg_options.num_people-1))) * 50
    res = abs(FNMR - FMR)
    m = min(res)
    for i in range(100):
        if m == res[i]:
            Sth = i

    f = open('./output/verification_output.txt', 'a')    
    ecg_options.print_options(f)
    print('Sth :', str(Sth),' FNMR :',str(FNMR[Sth]), ' FMR :',str(FMR[Sth]), file = f)
    print('\n', file = f)
    f.close()
    '''

    return

# In[ ]:


def identification_test(ecg_options):
    
    model_ = os.path.join('./model', str(ecg_options.model_name))
    train_dataset, train_target = preparing_ecg(ecg_options)
    
    kfold = KFold(n_splits=5, random_state=1, shuffle=True)
    len_data = len(train_dataset[0])
    
    machine_learning_list = ['GaussianNB' , 'LogisticRegression', 'KNN', 'RandomForestClassifier', 'SVC']
    accuracy_list = []

    if ecg_options.model_name != 'none':
        Train_dataset = train_dataset.reshape(len(train_dataset), len_data, 1)
        model_m = load_model(model_, compile = False)
        intermediate_layer_model = Model(inputs=model_m.input, outputs=model_m.layers[13].output)
        train_dataset = intermediate_layer_model.predict(Train_dataset)

    m1_nb = GaussianNB()
    accuracy1 = np.mean(cross_val_score(m1_nb, train_dataset, train_target, scoring="accuracy", cv=kfold))
    accuracy_list.append(accuracy1)

    m2_log = LogisticRegression(solver='newton-cg')
    accuracy2 = np.mean(cross_val_score(m2_log, train_dataset, train_target, scoring="accuracy", cv=kfold))
    accuracy_list.append(accuracy2)

    m3_knn_5 = KNeighborsClassifier(n_neighbors = 5)
    m3_knn_10 = KNeighborsClassifier(n_neighbors = 10)
    m3_knn_30 = KNeighborsClassifier(n_neighbors = 30)
    accuracy3 = np.max([np.mean(cross_val_score(m3_knn_5, train_dataset, train_target, scoring="accuracy", cv=kfold)),
                              np.mean(cross_val_score(m3_knn_10, train_dataset, train_target, scoring="accuracy", cv=kfold)),
                              np.mean(cross_val_score(m3_knn_30, train_dataset, train_target, scoring="accuracy", cv=kfold))])
    accuracy_list.append(accuracy3)

    m4_rf = RandomForestClassifier(n_estimators=100)
    accuracy4 = np.mean(cross_val_score(m4_rf, train_dataset, train_target, scoring="accuracy", cv=kfold))
    accuracy_list.append(accuracy4)
    
    m5_svc = SVC(gamma='scale')
    accuracy5 = np.mean(cross_val_score(m5_svc, train_dataset, train_target, scoring="accuracy", cv=kfold))
    accuracy_list.append(accuracy5)

    f = open('./output/identification_output.txt', 'a')
    ecg_options.print_options(f)
    for i in range(len(machine_learning_list)):
        print(machine_learning_list[i] + ' : ' +str(accuracy_list[i]), file = f)
    print('\n', file = f)
    f.close()
    

