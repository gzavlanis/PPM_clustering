import pandas as pd
from matplotlib import pyplot
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# import and plot the PPM data, that we will use for clustering
X= np.array(pd.read_csv(r'PPM-in.txt', delimiter= '\t', header= None))
y= np.array(pd.read_csv(r'PPM-out.txt', delimiter= '\t', header= None))
u= X[:, 1]
X[:, 1]= (u- u.min())/(u.max()- u.min()) # normalize data for Amplitude between 0 and 1
v= y[:, 1]
y[:, 1]= (v- v.min())/(v.max()- v.min()) # normalize data for Amplitude between 0 and 1
print(X)
print(y)

# plot input and output signal
pyplot.plot(X[:, 0], X[:, 1])
pyplot.xlim([0.2e-7, 0.3e-7])
pyplot.title('The PPM signal as produced from transmitter')
pyplot.xlabel('Time (Seconds)', fontsize= 16)
pyplot.ylabel('Amplitude (Volts)', fontsize= 16)
pyplot.grid()
pyplot.show()

pyplot.plot(y[:, 0], y[:, 1])
pyplot.xlim([0.2e-7, 0.3e-7])
pyplot.title('The PPM signal after 1km of transmission')
pyplot.xlabel('Time (Seconds)', fontsize= 16)
pyplot.ylabel('Amplitude (Volts)', fontsize= 16)
pyplot.grid()
pyplot.show()

# create and train the Minibatch K-Means model
model= MiniBatchKMeans(n_clusters= 2, batch_size= 50, verbose= 1)
model.fit(X, y)

# make denoising and plot the clean signal
yhat= model.predict(X)
print(yhat)
pyplot.plot(y[:, 0], yhat)
pyplot.title('Signal after denoising')
pyplot.xlabel('Time (Seconds)', fontsize= 16)
pyplot.ylabel('Amplitude (Volts)', fontsize= 16)
pyplot.xlim([0.2e-7, 0.3e-7])
pyplot.grid()
pyplot.show()

# plot the signal before and after denoising in one plot
pyplot.plot(y[:, 0], yhat)
pyplot.plot(y[:, 0], y[:, 1])
pyplot.xlim([0.2e-7, 0.3e-7])
pyplot.title('Signal with noise and after denoising')
pyplot.xlabel('Time (Seconds)', fontsize= 16)
pyplot.ylabel('Amplitude (Volts)', fontsize= 16)
pyplot.grid()
pyplot.show()
