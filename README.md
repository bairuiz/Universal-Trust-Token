"# Universal-Trust-Token" 

Steps to upload trained model or data:
=====
Please check file paths in:
Universal-Trust-Token/Backend/PythonServer/TrustCal.py
Universal-Trust-Token/MachineLearning/Models/LSTM/lconfig.py
On Local Laptop Terminal:
scp -r <folder> username@tcs_universal_tt.heinz.cmu.edu:<model or data path>
scp <files> username@tcs_universal_tt.heinz.cmu.edu:<model or data path>
Eg.
scp -r data username@tcs_universal_tt.heinz.cmu.edu:~xumeil/Python_Server/lstm_data/
or
scp train.csv username@tcs_universal_tt.heinz.cmu.edu:~xumeil/Python_Server/lstm_data/data/

Steps to run server:
=====
ssh username@tcs_universal_tt.heinz.cmu.edu

cd ~xumeil/Universal-Trust-Token/Backend/PythonServer/
python3 -W ignore PythonServer.py

Note: Only one server with port 5000 can be run on tcs_universal_tt.heinz.cmu.edu

Steps to run app:
=====
Check latest version and open download link on Android device:
https://github.com/bairuiz/Universal-Trust-Token/blob/master/NewsDetector.apk

Steps to run TestClient (Simulator for app's reply/request):
=====
git clone <repo>
Open Universal-Trust-Token/Backend/TestClient/ in Eclipse and run Client.java
Comment out socekt close to simulate client connection drop.
