#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import os
from keras.models import load_model
from keras.models import Model
from biosppy.signals import ecg


# In[58]:


class ECG_user_options:
    def __init__(self, Model_, Len_resample, Num_resample_per_data, Type_resample, Ft_type):
        self.model_ = Model_
        self.len_resample = Len_resample
        self.num_resample_per_data = Num_resample_per_data
        self.type_resample = Type_resample
        self.ft_type = Ft_type
        self.num_file = 0
        self.num_data_per_file = 0
        self.data_path = ''
        if Type_resample == 'peak':
            self.num_resample_per_file = num_peak_per_file
        elif Type_resample == 'cycle':
            self.num_resample_per_file = num_peak_per_file - 1


# In[59]:


model_path = './model/'
templates_path = './enrollment_templates'
model_name = 'ECG_CNN_Conv1D_500_5_peak_none_130_137'
options = model_name.split('_')

num_peak_per_file = 15
thr = 0.88

model_ = os.path.join(model_path, model_name)
model_m = load_model(model_, compile = False)
intermediate_layer_model = Model(inputs=model_m.input, outputs=model_m.layers[13].output)

ecg_options = ECG_user_options(
                        Model_ = intermediate_layer_model,
                        Len_resample = int(options[3]), 
                        Num_resample_per_data = int(options[4]), 
                        Type_resample = options[5], 
                        Ft_type = options[6]) 


# In[60]:


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

def filtering_detecting_peaks(data, measures):
    measure = {}
    #templates의 길이는 sampling rate의 60%
    out = ecg.ecg(signal=data.A, sampling_rate=1000, show = False )
    data['filtered'] = out['filtered']
    measure['peaklist'] = out['rpeaks']
    measure['ybeat'] = [out['filtered'][x] for x in out['rpeaks']]
    measures.append(measure)
    return

def resampling_ecg(data, measure, resamples):
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

def ECG_load_data(name_user):
    name_file = ['_{}.txt'.format(i+1) for i in range(ecg_options.num_file)]
    user_data_path = os.path.join(ecg_options.data_path, str(name_user))

    ecgdata = []
    for fname in name_file:
        path = user_data_path + fname
        data = pd.read_csv(path, delimiter = "\t")
        data = data.drop(data.columns[[0, 2, 3]], axis='columns')
        data.columns = ["A"]
        ecgdata.append(data)
    return ecgdata

def ECG_preprocessing(ecgdata):
    measures = []
    resamples = []
    
    for i in range(ecg_options.num_file):
        filtering_detecting_peaks(ecgdata[i], measures)
        resampling_ecg(ecgdata[i], measures[i],resamples)     
    
    num_data_per_file = ecg_options.num_data_per_file
    len_data = (ecg_options.num_resample_per_data) * (ecg_options.len_resample)
    len_resample = ecg_options.len_resample
    
    ECG_data = []
    
    for j in range(ecg_options.num_file):
        for k in range(num_data_per_file):
            if ecg_options.ft_type == 'stft':
                 f, t, data = signal.stft(resamples[j][(len_resample)*k : (len_resample)*k + len_data], 1000)
            elif ecg_options.ft_type == 'dct':
                data = scipy.fftpack.dct(resamples[j][(len_resample)*k : (len_resample)*k + len_data])
            elif ecg_options.ft_type == 'wavelet_1':
                (data1, data2) =  pywt.dwt(resamples[j][(len_resample)*k : (len_resample)*k + len_data], 'db1')
                data = np.concatenate((data1, data2), axis=0)
            elif ecg_options.ft_type == 'wavelet_2':
                (data1, data2) =  pywt.dwt(resamples[j][(len_resample)*k : (len_resample)*k + len_data], 'db1')
                data = data2
            elif ecg_options.ft_type == 'none':
                data = resamples[j][(len_resample)*k : (len_resample)*k + len_data]
            else:
                print('ft_type : 잘못된 입력입니다.')
                return
            ECG_data.append(data)
    ECG_data = np.array(ECG_data)
    ECG_data = ECG_data.reshape(len(ECG_data), len_data, 1)
    return ECG_data

def ECG_make_templates(name_user, type_template):
    if type_template == 'enroll':
        ecg_options.num_data_per_file =  ecg_options.num_resample_per_file - ecg_options.num_resample_per_data + 1
        ecg_options.num_file = 4
        ecg_options.data_path = './data_for_enrollment/'         
    elif type_template == 'verify':
        ecg_options.num_data_per_file = 7
        ecg_options.num_file = 1
        ecg_options.data_path = './data_for_verification/' 


    ecgdata = ECG_load_data(name_user)
    ECG_data = ECG_preprocessing(ecgdata)
    
    ECG_template = (ecg_options.model_).predict(ECG_data)
    len_template = len(ECG_template[0])

    if type_template == 'enroll':
        enrollment_template = np.zeros((len_template))
        validation_template = np.zeros((len_template))
        for i in range(ecg_options.num_data_per_file * (ecg_options.num_file-1)):
            enrollment_template += ECG_template[i]
        for i in range(ecg_options.num_data_per_file * (ecg_options.num_file-1), ecg_options.num_data_per_file * ecg_options.num_file):
            validation_template += ECG_template[i]
        
        enrollment_template /= ecg_options.num_data_per_file * (ecg_options.num_file-1)
        validation_template /= ecg_options.num_data_per_file
        return enrollment_template, validation_template
    
    elif type_template == 'verify':
        verification_template = np.zeros((len_template))
        
        for i in range(ecg_options.num_data_per_file * ecg_options.num_file):
            verification_template += ECG_template[i]
        verification_template /= ecg_options.num_data_per_file * ecg_options.num_file
        return verification_template

def ECG_user_enrollment(user_name):
    user_enrollment_template, user_validation_template = ECG_make_templates(user_name, 'enroll')
    val = cal_threshold(user_enrollment_template, user_validation_template, 'corr')
    print(val)
    if val > thr:
        template_name = os.path.join(templates_path, user_name)
        np.save(template_name, user_enrollment_template)
        print('등록 성공')
        return True
    else:
        print('등록 실패')
        return False
 
    
def ECG_user_verification(user_name):
    user_verification_template = ECG_make_templates(user_name, 'verify')
    enrollment_template_name = os.path.join(templates_path, user_name + '.npy')
    user_enrollment_template = np.load(enrollment_template_name)
    val = cal_threshold(user_enrollment_template, user_verification_template, 'corr')
    print(val)
    if val > thr:
        print('인증 성공')
        return True
    else:
        print('인증 실패')
        return False


# In[65]:


ECG_user_enrollment('21')


# In[66]:


ECG_user_verification('21')


# In[ ]:




