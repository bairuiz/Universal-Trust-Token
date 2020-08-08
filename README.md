Steps to check about trained model or data:
=====
- Please check file paths in:
```
Universal-Trust-Token/Server/PythonServer/TrustCal.py
Universal-Trust-Token/MachineLearning/Models/LSTM/lconfig.py
```

Steps to run server:
=====
```
Go to Universal-Trust-Token/Server/PythonServer/Models/ and unzip newData_w_title.csv.zip
run the following line to create the vectors:

python3 vectors.py

python3 -W ignore PythonServer.py

This will start the python server which calls the other modules too.
```
- Issue ctrl+c interrupt to close server.

Steps to run app:
=====
Check and download [latest version](https://github.com/bairuiz/Universal-Trust-Token/blob/master/NewsDetector.apk) on Android device.

Steps to run TestClient (Simulator for app's reply/request):
=====
```
git clone <repo>
```
- Incase of running a Test Client on Java.
- Open Universal-Trust-Token/Server/TestClient/ in Eclipse and run Client.java
- Comment out socket close to simulate client connection drop.

Description of files/folders:

Folder: Universal-Trust-Token/Server/PythonServer/Models/
- This folder contains all the pre-trained models that are called by our server to predict the news articale passed by our application.
- This folder also contains our data which is stored in a zip format.
- In this folder we also have to run the vectors.py file to create the vectors that are required for our server to run.

Folder: Universal-Trust-Token/MachineLearning/
- This folder contains all the final codes that we used for our different machine learning models.
- They use and create the trained models that we have used in our server.
- It also contains code that we use to prepare our dataset and the feature engineering that we used.

Folder: Universal-Trust-Token/NewsDetector/
- This folder contains the code that we used for creating our Android Application.
- You can also find the apk for the application on the main page as mentioned above.

Folder: Universal-Trust-Token/MachineLearning/Models/LSTM/
- This folder is used to store the information required to run the LSTM Model that is used.
- Folder structure should be maintained as it is coded and used in our server the same way.
