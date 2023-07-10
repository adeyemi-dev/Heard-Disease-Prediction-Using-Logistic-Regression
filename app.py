import numpy as np
import pickle 
import streamlit as st 


class Logistic_Regression():
  
  # declaring learning rate and number of iteration(Hyperparameters)
  def __init__(self,learning_rate,no_of_iteration):

    self.learning_rate = learning_rate
    self.no_of_iteration = no_of_iteration


  # fit the function to train our model with the dataset 
  def fit(self,X,Y):

    # no of data points in the dataset (number of rows) ==> m
    # no of input features in the dataset (number of columns )==> n
    self.m , self.n = X.shape

    #initiate the weight and bais value 
    self.w = np.zeros(self.n)
    self.b = 0

    self.X = X
    self.Y = Y

    # implementing gradient descent 
    for i in range(self.no_of_iteration):
      self.update_weight()


  def update_weight(self):

    # we need the formular for y_hat first (sigma function)

    Y_hat = 1 / (1 + np.exp( - (self.X.dot(self.w) + self.b) ))  # wx + b

    # for the derivatives

    dw = (1/self.m)*np.dot(self.X.T, (Y_hat - self.Y)) # taking the transpose number of column of one matrix should match no of row in the next
                                                       # X = [769 x 8]  Y = [769 x 1]

    db = (1/self.m)*np.sum(Y_hat - self.Y)
    
    # updating the weight and bais using gradient descent equation

    self.w = self.w - self.learning_rate * dw

    self.b = self.b - self.learning_rate * db

    # Sigmoid Equation & Decision Boundary

  def predict(self, X):  
    
    Y_pred =  1 / (1 + np.exp( - (X.dot(self.w) + self.b) )) 
    Y_pred = np.where(Y_pred > 0.5 , 1 , 0)
    return Y_pred


#loading the saved model python -m pickletools -a file.pkl
loaded_model = pickle.load(open('trained_model.pkl','rb'))

#creating the prediction function
def heart_prediction(*input_data):

  # change the input data to numpy array before we can make prediction on it
  print(f"Args: {input_data}")
  input_data_as_numpy_array = np.asarray(input_data)

  # reshape the numpy array as we are predicting for only one instance 
  input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
  prediction = loaded_model.predict(input_data_reshaped)
  print(f"Prediction: {prediction}")

  if (prediction[0] == 0):
    return 'The person does not have Heart Disease'
  else:
    return 'The person has Heart Disease' 

def main():

  #title of our web page 
  st.title('Heart Disease prediction Web App')

  #Getting the imput data from the user
  Age = st.number_input('Age', min_value=20, max_value=90 ,help ='Paient Age')
  Sex = st.selectbox('Gender', [0, 1],format_func=lambda x: 'Female' if x == 0 else 'Male',help='(0 = Female, 1 = Male)')
  st.write('Selected Gender:', Sex)
  Chest_pain_type  = st.selectbox(
    'Chest Pain type?',
    [1,2,3,4], format_func=lambda x: 'Typical angina' if x == 1 else 'Atypical angina' if x == 2 else 'Non-anginal pain' if x == 3 else 'Asymptomatic',help='(1 = Typical angina refers to chest pain caused by reduced blood flow to the heart muscle , 2 = Atypical angina refers to chest pain that is not typical of angina , 3 = Non-anginal pain refers to chest pain that is not caused by reduced blood flow to the heart , 4 = Asymptomatic means the patient does not experience chest pain.)')
  st.write('You selected:', Chest_pain_type)
  BP = st.number_input('Resting Blood Pressure (mm Hg)', min_value=90, max_value=200, help ='The resting blood pressure of the patient, measured in millimeters of mercury (mm Hg).')
  Cholesterol = st.number_input('Serum Cholesterol (mg/dl)', min_value=120, max_value=300 , help='The serum cholesterol level of the patient, measured in milligrams per deciliter (mg/dL). Cholesterol is a fatty substance found in the blood and high levels can be associated with heart disease')
  FBS_over_120 = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1] ,help='(0 = False, 1 = True)')
  EKG_results = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2], help='(0 = Normal indicates a normal ECG reading , 1 =  ST-T wave abnormality refers to abnormalities in the ST segment or T wave of the ECG , 2 = Left ventricular hypertrophy suggests an enlargement of the left ventricle of the heart)')
  Max_HR = st.number_input('Maximum Heart Rate', min_value=70, max_value=220, help=' The maximum heart rate achieved by the patient during exercise.')
  Exercise_angina = st.selectbox('Exercise Induced Angina', [0, 1], help='Indicates whether the patient experienced exercise-induced angina.(0 = False , 1 = True)')
  ST_depression = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=7.0, step=0.1 , help='ST depression induced by exercise relative to rest. ST depression is measured in millimeters on an electrocardiogram (ECG) and represents abnormalities in the hearts electrical activity')
  Scope_of_ST = st.selectbox('Slope of the Peak Exercise ST Segment on the ECG', [1, 2, 3], help = ' 1 = Upsloping indicates the ST segment has a positive slope during exercise , 2 = Flat indicates the ST segment is relatively horizontal during exercise , 3 = Downsloping indicates the ST segment has a negative slope during exercise' )
  Number_of_vessels_fluro = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3], help='The number of major vessels (0-4) colored by fluoroscopy. Fluoroscopy is a medical imaging technique that uses a contrast agent to visualize blood vessels.')
  Thallium = st.selectbox('Thallium stress test', [3, 6, 7], help = '3 = Normal indicates a normal thallium stress test result , 6 = Fixed defect suggests a permanent defect in blood flow to a certain area of the heart , 7 = Reversible defect indicates a temporary defect in blood flow to a certain area of the heart')

  #code for prediction
  diagnosis = ''

  #creating a button for prediction

  if st.button("Heart Disese test result"):
    diagnosis = heart_prediction(Age,Sex,Chest_pain_type,BP,Cholesterol,FBS_over_120,EKG_results,Max_HR,Exercise_angina,ST_depression,Scope_of_ST,Number_of_vessels_fluro,Thallium)

  st.success(diagnosis)


if __name__ == '__main__':
  main()



