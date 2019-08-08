#  Gradient class activation maps DDD

visualize Gradient class activation maps of network trained for driver distraction detection using StateFarm database. 



######################### INFORMATION #################################


This repository consists of CNN visualization of intermediate layers of a driver distraction detection using StateFarm database. Two models were trained with similar structure and are visualised here. One that classifies only mobile based distraction and another that classifies all distractions.

Trained model and corresponding files are in : https://github.com/n-akram/DriverDistractionDetection

Classes used by model classifying only mobile based distractions: C0 to C4

Implementation from: https://github.com/jacobgil/keras-grad-cam

Detailed information: https://arxiv.org/pdf/1610.02391v1.pdf


Database used: StateFarm driver distraction detection.

https://www.kaggle.com/c/state-farm-distracted-driver-detection/data


#################### Technical INFORMATION ##############################

Implemented using: forVis environment

Activate using : source forVis/bin/activate


Environment details:
Python version: 3.6.6
Tensorflow backend : 1.14.0
Keras : 2.2.4
Open CV : 4.1.0

System configuration:
OS: Windows 10
CPU: Intel core i7

############################ To Run #####################################

1. Use TensorKeras environment. Activate using: 
    PS : . ..\PythonEnvs\TensorKeras\Scripts\activate.ps1

2. Install the relevant packages from "requirements.txt"
    pip install -r requirements.txt

3. Run "python main.py" : for grad cam of model trained on all classes using training image
    Note: add the appropriate model to be tested in the folder 'sampleModel'

5. Run "python main.py -m": for grad cam of model trained on mobile classes using training image

6. Run "python main.py -t" : for grad cam of model trained on all classes using test image

7. Run "python main.py -m -t" : for grad cam of model trained on mobile classes using test image