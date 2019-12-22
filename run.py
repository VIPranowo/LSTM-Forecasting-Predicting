__author__ = "Vicko Pranowo" #(modified and edited)
__copyright__ = "Vicko Pranowo 2019"
__version__ = "3.0.0"
__license__ = "UGM_AI_CENTER"

import os
import json
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model
import csv

from pandas import DataFrame

import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

listtahun=[]
for i in range (1961,2015):
    listtahun.append(str(i))

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

def plot_results(predicted_data, true_data,year_data,name_legend):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(year_data, true_data, label='True Data')
    plt.plot(year_data,predicted_data, label='Prediction')
    plt.title(name_legend)
    plt.legend()
    plt.show()

def plot_results_onlypredicted(predicted_data,year_axis,name_legend):
    fig= plt.figure(facecolor='white')
    plt.plot(predicted_data,year_axis,label='prediction')
    plt.title(name_legend)
    plt.legend()
    plt.show()


# def plot_results_multiple(predicted_data, true_data, prediction_len):
#     fig = plt.figure(facecolor='white')
#     ax = fig.add_subplot(111)
#     ax.plot(true_data, label='True Data')
# 	# Pad the list of predictions to shift it in the graph to it's correct start
#     for i, data in enumerate(predicted_data):
#         padding = [None for p in range(i * prediction_len)]
#         plt.plot(padding + data, label='Prediction')
#         plt.legend()
#     plt.show()

def main():
    configs = json.load(open('configcrops.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    
    namaefile= configs['data']['filename1']

    with open(namaefile,'r') as dataframe:
        hasil=json.load(dataframe)
    # print(hasil)

    temp=[]
    listhasil=[]
    for key,value in hasil.items():
        temp= [key,value]
        listhasil.append(temp)

    listkota= ['Kulon Progo','Bantul','Gunung Kidul','Sleman','DIY','Bandung','Tasikmalaya','Majalengka','Cirebon','Kuningan','Garut','Sumedang','Cianjur','Subang','Purwakarta','Indramayu','Ciamis','Sukabumi','Bogor','Bekasi','Karawang']
    kodekota= ['KLP','BTL','GKD','SLM','DIY','BD','TKM','MJK','CRB','KNG','GRT','SMD','CJR','SBG','PWK','IDY','CMS','SKB','BGR','BKS','KRW']
    listtahun=[]
    for i in range (1961,2015):
        listtahun.append(str(i))

    data=[]
    #listhasil
    #=[["Kulon Progo",{data tahun dan crops},"DIY",{data tahun dan crops}]
    datacrops=[]
    datalengkapcrops=[]
    datalengkaptahun=[]
    datatahun_semuadaerah=[]

    #Variabel untuk tampung data csv
    kota_untuk_csv=[]       #pembuatan kolom kota pada csv
    kode_untuk_csv=[]       #pembuatan kolom kode untuk csv
    tahun_untuk_csv=[]      #pembuatan kolom tahun untuk csv
    crops_untuk_csv=[]      #pembuatan kolom crops untuk csv
    RMSE_untuk_csv=[]

    semua_data_csv=[]       #untuk menampung data kota,kode,tahun, dan crops beserta rmse pada sebaris (tidak digunakan #cadangan)

    #Pengulangan compiling per kota
    for j in range (len(listkota)):
        if(len(listhasil[j][1])!=len(listtahun)):
            jlhprediksi=len(listtahun)-len(hasil[listkota[j]])+ (6)-1  #prediksi sampai 2020
        else:
            jlhprediksi=(6)-1   #prediksi sampai 2020

        datatahun_crops=listhasil[j][1]      #dapat data json crops dan tahun
        datatahun= list(datatahun_crops.keys())    #dapat data tahun pada satu daerah
        datatahunint= [int(x) for x in datatahun]   #konversi ke integer

        arraytahun=np.array(datatahunint)       #dibuat jadi array
        sorttahun= np.sort(arraytahun)          #sort dalam bentuk array
        datatahun_daerah=list(sorttahun)        #buat lagi jadi list
        datalengkaptahun.append(datatahun_daerah)

        for n in range (len(listhasil[j][1])):   #listhasil[j][1] = data tahun dan crops
            datacrops.append(float(listhasil[j][1][str(datatahun_daerah[n])]))
        datalengkapcrops.append(datacrops)
        datacrops=[]

    # print(datalengkapcrops)
    # print(datalengkaptahun)
    listcrops_daerah=[]
    hasillistcrops_daerah=[]

    for i in range(len(listkota)):
        for j in range(len(datalengkapcrops[i])):
            listcrops_daerah.append([datalengkapcrops[i][j]])    #pecah per tahun crops dalam satu list
        hasillistcrops_daerah.append(listcrops_daerah)
        listcrops_daerah=[]          

    # for i in range(len(listkota)):
    #     arraycrops_semua=np.array(hasillistcrops_daerah[i])
        # print(arraycrops_semua[0:20])

    arraycrops_semua=np.array(hasillistcrops_daerah)

    for i in range(len(listkota)):
        data = DataLoader(
            np.array(arraycrops_semua[i]),
            configs['data']['train_test_split']
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

        for ulang in range (len(datalengkaptahun[i])-len(predictions_point)):
            datalengkaptahun[i].remove(datalengkaptahun[i][ulang])              #for equality number of ground truth and prediction

        # Use the plot when you want to see the data graphically    
        # plot_results(predictions_point, y_test,datalengkaptahun[i],listkota[i])

        groundtrue= data._groundtruths(1)
        groundtrue=(groundtrue.ravel())
        # print(len(groundtrue))

        #Measure the RMSE
        RMSElist=[]
        for k in range(len(predictions_point)):
            errorrate=groundtrue[k+ulang]-predictions_point[k]
            hasilkuadrat=errorrate*errorrate
            RMSElist.append(hasilkuadrat)
        RMSE=sum(RMSElist)/(len(predictions_point))
        RMSE=RMSE**(1/2)        
        # print(RMSE)

        getdataforecast=data._forecasting(jlhprediksi,jlhprediksi)
        # print(len(getdataforecast))

        total_prediksi=jlhprediksi
        takefrom=jlhprediksi
        forecast_result=model.forecast(total_prediksi,getdataforecast,takefrom)
        # print(len(forecast_result))
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
        
        #make the number of predictions is equal to the number of the ground truth
        hasilprediksi=[]
        hasilprediksi.append(groundtrue[-(ulang+1):-ulang])
        hasilprediksi.append(groundtrue[-ulang:])

        for j in range(total_prediksi):
            getxlastnumber=array(forecast_result[(-n_steps-1):-1])
            x_input = getxlastnumber
            # print(x_input)

            x_input = x_input.reshape((1, n_steps, n_features))
            yhat = model.predict(x_input, verbose=0)
            # print(yhat[0][0])

            hasilprediksi.append(yhat[0])                       #untuk dikirimkan ke json
            forecast_result=np.append(forecast_result,yhat[0])  #untuk training forecast
            groundtrue=np.append(groundtrue,yhat[0])            #untuk plotting ke grafik

            # print(len(groundtrue))
            # prediction_point=np.append(prediction_point,yhat[0])
        # print(hasilprediksi)      #hasilprediksi dalam bentuk array, hasilprediksi[0] dalam bntk list, hasilprediksi[0][0] dalam bentuk skalar

        semuatahun=datalengkaptahun[i]
        tahunbaru=[]
        terakhirtahun=datalengkaptahun[i][len(datalengkaptahun[i])-1]

        rangetahun_input=len(groundtrue)-len(datalengkaptahun[i])
        # print(rangetahun_input)

        if(len(datalengkaptahun[i])<len(groundtrue)):
            for z in range (rangetahun_input):
                semuatahun.append(terakhirtahun)        #untuk grafik
                tahunbaru.append(terakhirtahun)         #untuk dikirimkan ke json
                terakhirtahun=terakhirtahun+1
        # print(tahunbaru)


        # Use the plot when you want to see the data graphically
        # plot_results_onlypredicted(semuatahun,groundtrue,listkota[i])

        #To check the length of ground true is equal to the datalengkaptahun[i] or the number of years record at specific Entity
        # print(len(groundtrue))
        # print(len(datalengkaptahun[i]))

        # semuahasil_csv=[]
        # csv_data_kota=duplikathasil.get(column).values[:]

        #To record all data into LIST to make CSV
        for jlh in range(len(semuatahun)):
            kota_untuk_csv.append(listkota[i])
            kode_untuk_csv.append(kodekota[i])
            RMSE_untuk_csv.append(RMSE[0])
            tahun_untuk_csv.append(semuatahun[jlh])
            crops_untuk_csv.append(groundtrue[jlh])

        #Alternative solution for csv
        # for jlh in range(len(datalengkapcrops[i])):
        #     kota_untuk_csv.append(listkota[i])
        #     kode_untuk_csv.append(kodekota[i])
        #     # tahun_untuk_csv.append(datalengkaptahun[i][jlh])
        #     # crops_untuk_csv.append(datalengkapcrops[i][jlh])
        #     RMSE_untuk_csv.append(RMSE[0])

        # for jlh in range (rangetahun_input):
        #     kota_untuk_csv.append(listkota[i])
        #     kode_untuk_csv.append(kodekota[i])
        #     # tahun_untuk_csv.append(tahunbaru[jlh])
        #     # crops_untuk_csv.append(hasilprediksi[jlh][0])
        #     RMSE_untuk_csv.append(RMSE[0])

        HasilCSV={'Entity':kota_untuk_csv,
                  'Code':kode_untuk_csv,
                  'Year':tahun_untuk_csv,
                  ' crop(tonnes per hectare)':crops_untuk_csv,
                  'RMSE':RMSE_untuk_csv
                  }
        df = DataFrame(HasilCSV, columns= ['Entity', 'Code','Year',' crop(tonnes per hectare)','RMSE'])
        print(df)


        filebaca_csv=configs["data"]["newcsv"]
        filebaca_csv1=configs["data"]["newcsv1"]


        #name of data export can be changed through configcrops.json (variable filebaca_csv and filebacacsv1 only for explanation)
        export_csv=df.to_csv(r'/home/biovick/Downloads/tkte/sudiro/Forecasting-and-Predicting Crops into Visualization/data/newTomatov2.csv',index=False)

if __name__ == '__main__':
    main()
