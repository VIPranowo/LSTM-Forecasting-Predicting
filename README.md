# Forecasting and predicting Crops using LSTM (Long Short Term Memory)
The task is to predict the crops from 1961 until 2014 and to forecast the number of crops in the following year until 2020 using Machine Learning method of LSTM (Long Short Term Memory). On the other task, the data can be seen from data/Maize.json and data/Tomato.json not fully provided from the year 1961 to 2014. The crops are predicted for West Java Province and Special Region of Yogyakarta which consecutively the number of entities are 16 and 5.

List of Special Region of Yogyakarta:
1. kulon progo
2. bantul
3. gunung kidul
4. sleman
5. DIY

List of West Java Province:
1. Bandung
2. Tasikmalaya
3. Majalengka
4. Cirebon
5. Kuningan 
6. Garut
7. Sumedang
8. Cianjur
9. Subang
10. Purwakarta
11. Indramayu
12. Ciamis
13. Sukabumi
14. Bogor
15. Bekasi
16. Karawang

## Requirement:
Install requirement.txt to compute the coding without problem (Under development):
1. Python 3.7.2
2. Matplotlib 3.1.2
3. Pip 3.7
4. Keras 2.3.1
5. Pandas 0.25.3

## Next development:
Using Tensorflow 1.15 as suggested

## How to Run:
- Get into the environment of python 3.7.x
- python3.7 run.py

## Result:
The result will be seen as the following: 
- The graphics show the prediction crops data of all entities in two provinces
- The graphics show the following year or inadequate data of crops until the data shown year of 2020
- The data can be shown in newTomato.csv and newMaize.csv , then visualized by Sudiro
- Substract json data from Sudiro and Thea to manipulate data into LSTM method to predict and forecast the next step

(Under development for the image result):

The visualization will be depicted in this link of assignment 
(Sudiro, link: https://github.com/sudiroeen/Web-Based-Visualization-of-Agricultural-Data)

Another proposed method by:
(Thea Kirana proposed a method of linear regression and polynomial regression data crops)

## Thanks to:
1. Jakob Aungiers (link: https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction)
2. Jason Brownlee (link: https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/)
3. Python Tutorial (link: https://datatofish.com/export-dataframe-to-csv/)

## Modified by:
Vicko Pranowo

## Supported by:
1. Mr. Igi Ardiyanto (UGM AI Center)
2. Thea Kirana
3. Sudiro


