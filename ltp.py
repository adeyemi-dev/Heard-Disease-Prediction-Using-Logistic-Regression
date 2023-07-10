import numpy as py 
import pickle 
loaded_model = pickle.load(open('trained_model.pkl','rb'))

input_data = (55,0,1,120,244,0,1,162,0,1.1,2,0,2)

# change the input data to numpy array before we can make prediction on it
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = loaded_model.predict(input_data_reshaped)

print(prediction)

if (prediction[0] == 0):
  print('The person does not have Heart Disease')
else:
  print('The person has Heart Disease')  