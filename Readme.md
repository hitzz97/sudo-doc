details about this directory:
_______________________________________________________________________________________
dataset:
contains the image dataset required for the training of the
machine learning algorithm to predict skin diseases

templates:
contains the html pages flask serves for the interfacing
of this application 

uploads:
the images uploaded by the user are stored here 

app3.py:
the Flask server or the main python file.

manifest.yml:
should contain the app name you are using on the platform like 
ibm cloud and must mention the upper limits of the memory in this 
file carefully.

procfile:
this fill contains the name of the app to be run only for 
cloud like ibm in this case it contains the name app3.py

requirements.txt
this contains the list of required modules to run the script
with native pthon installation 

symp.csv.xlsx:
this is the dataset for symptoms used for data science in the project.
______________________________________________________________________________________

To Run the app on local machin just clone the repo and 
run the app3.py.
_______________________________________________________________________________________
INPUT FORMAT IN THE TEXT AREA ON APPLICATION:
write the symptoms comma seperated without an space. Eg:fever,pain chest,headache,cough

