Steps to upload trained model or data:
=====
- Please check file paths in:
```
Universal-Trust-Token/Server/PythonServer/TrustCal.py
Universal-Trust-Token/MachineLearning/Models/LSTM/lconfig.py
```
- On Local Laptop Terminal:
```
scp -r <folder> username@tcs_universal_tt.heinz.cmu.edu:<model or data path>
scp <files> username@tcs_universal_tt.heinz.cmu.edu:<model or data path>
```
- For example:
```
scp -r data username@tcs_universal_tt.heinz.cmu.edu:~xumeil/Python_Server/lstm_data/
or
scp train.csv username@tcs_universal_tt.heinz.cmu.edu:~xumeil/Python_Server/lstm_data/data/
```

Steps to run server:
=====
```
ssh username@tcs_universal_tt.heinz.cmu.edu

cd ~xumeil/Universal-Trust-Token/Server/PythonServer/

go to Universal-Trust-Token/Server/PythonServer/Models/ and unzip newData_w_title.csv.zip
run the following line to create the vectors:
python3 vectors.py

python3 -W ignore PythonServer.py

or

python3 -W ignore ~xumeil/Universal-Trust-Token/Server/PythonServer/PythonServer.py
```
- Issue ctrl+c interrupt to close server.

*Note: Only one server with port 5000 can be run on tcs_universal_tt.heinz.cmu.edu*

Steps to run app:
=====
Check and download [latest version](https://github.com/bairuiz/Universal-Trust-Token/blob/master/NewsDetector.apk) on Android device.

Steps to run TestClient (Simulator for app's reply/request):
=====
```
git clone <repo>
```
- Open Universal-Trust-Token/Server/TestClient/ in Eclipse and run Client.java
- Comment out socekt close to simulate client connection drop.
