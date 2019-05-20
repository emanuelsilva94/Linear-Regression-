print(__doc__)

from sklearn import preprocessing
import numpy as np

def scale_data ( dataset_data, type_of_scale ):
    if type_of_scale == 'Standard':
        # Create the standard scaler
        scaler = preprocessing.StandardScaler().fit(dataset_data)
        # Transform dataset data
        scaled_data = scaler.transform(dataset_data)
        # Return scaled data
        return np.array(scaled_data)
    else:
        # Create the min max scaler
        scaler = preprocessing.MinMaxScaler()
        # Transform dataset data
        scaled_data = scaler.fit_transform(dataset_data)
        # Return scaled data
        return np.array(scaled_data)