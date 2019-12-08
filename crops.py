__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model

import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
 
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_onlypredicted(predicted_data):
    fig= plt.figure(facecolor='white')
    plt.plot(predicted_data,label='prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def main():
    configs = json.load(open('configcrops.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    model = Model()
    model.build_model(configs)
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    
	# in-memory training
    model.train(
        x,
        y,
        epochs = configs['training']['epochs'],
        batch_size = configs['training']['batch_size'],
        save_dir = configs['model']['save_dir']
    )

# Yogyakarta: Kulon progo, bantul, gunung kidul, sleman, DIY
# Jawa Barat: Bandung, Tasikmalaya, Majalengka, Cirebon, Kuningan, Garut, Sumedang, Cianjut, Subang, Purwakarta, Indramayu
# Ciamis, Sukabumi, Bogor, Bekasi, Karawang


    # # out-of memory generative training
    # steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    # model.train_generator(
    #     data_gen=data.generate_train_batch(
    #         seq_len=configs['data']['sequence_length'],
    #         batch_size=configs['training']['batch_size'],
    #         normalise=configs['data']['normalise']
    #     ),
    #     epochs=configs['training']['epochs'],
    #     batch_size=configs['training']['batch_size'],
    #     steps_per_epoch=steps_per_epoch,
    #     save_dir=configs['model']['save_dir']
    # )

	# # save_dir = configs['model']['save_dir']
	


    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # print(x_test)
    # print(y_test)

    # predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
   
    predictions_point = model.predict_point_by_point(x_test)
    print(len(predictions_point))
    plot_results(predictions_point, y_test)

    # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
    # predictions_full = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    # plot_results(predictions_full, y_test)

    groundtrue= data._groundtruths(1)
    groundtrue=(groundtrue.ravel())
    print(len(groundtrue))

    RMSElist=[]
    for i in range(len(groundtrue)):
        errorrate=groundtrue[i]-predictions_point[i]
        hasilkuadrat=errorrate*errorrate
        RMSElist.append(hasilkuadrat)
    RMSE=sum(RMSElist)/(len(predictions_point)-2)
    RMSE=RMSE**(1/2)        
    print(RMSE)

    getdataforecast=data._forecasting(5,1)

    total_prediksi=5
    takefrom=5
    forecast_result=model.forecast(total_prediksi,getdataforecast,takefrom)
    # print(forecast_result[0])
    # forecast_result=np.append(forecast_result,[0.0])
    # print(forecast_result)
    
    n_steps = 8
    # split into samples
    X, y = split_sequence(forecast_result, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    # print(X)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=200, verbose=0)
    
    # demonstrate prediction
    for j in range(total_prediksi):
        getxlastnumber=array(forecast_result[(-n_steps-1):-1])
        x_input = getxlastnumber
        # print(x_input)

        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        # print(yhat[0][0])

        forecast_result=np.append(forecast_result,yhat[0])
        # prediction_point=np.append(prediction_point,yhat[0])
    
    plot_results_onlypredicted(forecast_result)

    # forecast_test= model.forecast(total_prediksi,x_test,takefrom)
    # plot_results(forecast_test, y_test)

if __name__ == '__main__':
    main()