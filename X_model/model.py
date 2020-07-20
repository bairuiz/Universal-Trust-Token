# model.py
import numpy as np
import time

# Common
modelParamFileName = "tmp.npy"

# Predict Function
def predict(url):
    xx = np.load(modelParamFileName)
    return xx

# Train Script
if __name__ == "__main__":
    print("simulating of training delay")
    for i in range(3):
        print("Training... %d" % i)
        time.sleep(1)
    x = np.arange(10)
    np.save(modelParamFileName, x);
    print("Done Training!")
    