from flask import Flask, render_template, redirect, url_for, request, jsonify,Response
from werkzeug import secure_filename
import os
import threading
import random
import pickle
import cv2
import numpy as np

#this command only for linux users 
os.system("chmod -R 777 dataset")
# to maintain and keep track of what to return as html back to the frontend.

NUM,POS=1,20
RETURN='<div class="received" style="border:10px solid transparent;width:49%;background-color:#e4ffce;border-radius:45%;text-align:center;float: left;font-size: 20px;" >'
RETURN2='<br><a href="https://www.google.com/search?client=chrome-b-d&q= " target="_blank">-Know more-</a></div>'

#variables to sync the threads of server and the predicter and 
#also to define which thread will respond "IMG" = 1 means image predictor 

output=False
Input=False
Out,In="",""
IMG=0

#path of the directory the app is reciding in.

try:
	print(os.listdir(os.path.join("\\".join(os.path.abspath(__file__).split("\\")[:-1]))))
except Exception as e:
	print(e)
			 
'''making the one time cache to keep track of the diseases and its symptoms'''
'''code for the disease predictor given the symptoms begins from here'''
def make_cache():
	import pandas as pd
	import pickle 
	import os
	'''the file that contains the data set for the diseases'''
	f=pd.read_excel(os.path.join("\\".join(os.path.abspath(__file__).split("\\")[:-1]),"symp.csv.xlsx"))

	f["diseases"].fillna(value="0",inplace=True)
	f["count"].fillna(value="0",inplace=True)

	print("Starting....")

	dise={}
	symp={}

	count=-1
	for i in range(1864):
		if f["diseases"][i]!='0':
			count+=1
			dise[count]=[f["diseases"][i].split("_")[1],count]
			symp[count]=[]  
		
		try:
			a=f["symp"][i]
			a=a.split("_")
			if len(a)>2:
				symp[count].append(a[-1])
				symp[count].append(a[1].split("^")[0])
			else:
				symp[count].append(a[-1])
		except:
			symp[count].append(a) 
		symp[count]=list(set(symp[count]))

	''' saving the model'''
	print ("saving to","\\".join(os.path.abspath(__file__).split("\\")[:-1]))
	pickle_out=open(os.path.join("\\".join(os.path.abspath(__file__).split("\\")[:-1]),"dise.pickle"),"wb")
	pickle.dump(dise,pickle_out)
	pickle_out.close()

	pickle_out=open(os.path.join("\\".join(os.path.abspath(__file__).split("\\")[:-1]),"symp.pickle"),"wb")
	pickle.dump(symp,pickle_out)
	pickle_out.close()
	
'''loading the resuts saved by symptoms.py file'''
	
dise=""
symp=""
try: 
	dise=pickle.load(open(os.path.join("\\".join(os.path.abspath(__file__).split("\\")[:-1]),"dise.pickle"),"rb"))
	symp=pickle.load(open(os.path.join("\\".join(os.path.abspath(__file__).split("\\")[:-1]),"symp.pickle"),"rb"))
except:
	print("DISEASES,SYMPTOMS Cache <Not Found>\nBUILDING Cache")
	make_cache()
	dise=pickle.load(open(os.path.join("\\".join(os.path.abspath(__file__).split("\\")[:-1]),"dise.pickle"),"rb"))
	symp=pickle.load(open(os.path.join("\\".join(os.path.abspath(__file__).split("\\")[:-1]),"symp.pickle"),"rb"))
	print("\n\n[done]\n")
	
def check(a,b):
    count=0
    for i in a:
        if i in b:
            count+=1
    return count
'''function to resolve the collision between one or more diseases with same number of similar symptoms 
    based on the user inputs of extra symptoms which he feels and are unique to particular disease he must 
    be sufering from that.'''
def collision_solver(ans,inp):
    global Input,In,output,Out,NUM,RETURN,RETURN2
    space=[]
    for i in ans:
        for symptom in symp[i[1]]:
            if symptom in space:
                space.remove(symptom)
            elif symptom not in inp:
                space.append(symptom)
    if not space:
        return ans
    random.shuffle(space)
    symptom=""
    for i in space:
        R=RETURN[:20]+'d'*NUM+'"'+RETURN[21:]
        NUM+=1
        Out=R+"Do you feel:- "+i+RETURN2[:63]+i+RETURN2[64:]  # to create the suggestion link to google
        print(Out)
        output=True
        Input=False
        ''' inpuut must be yes or no'''
        while not Input:pass
        a=In

        if a=="yes":
            symptom=i
            break
    for i in ans:
        if symptom in symp[i[1]]:
            return [i]

'''function to find the disease with maximum similarity with the symptoms '''
def comp(inp):
    global Input,In,output,Out,NUM,RETURN,RETURN2
    inp=inp.split(",")
    m=0
    ans=[]
    for i in symp:
        x=check(symp[i],inp)
        if x>m:
            ans=[dise[i]]
            m=x
        elif x==m:
            ans.append(dise[i])
    if len(ans)>1:
        ans=collision_solver(ans,inp)
        Out=""
        for i in ans:
            Out+=i[0]+","
        R=RETURN[:20]+'d'*NUM+'"'+RETURN[21:]
        NUM+=1
        Out=R+"you may be suffering from:"+Out[:-1]+RETURN2[:63]+Out[:-1]+RETURN2[64:] 
        output=True
        Input=False
    else:
        R=RETURN[:20]+'d'*NUM+'"'+RETURN[21:]
        NUM+=1
        Out=R+"you may be suffering from: "+ans[0][0]+RETURN2[:63]+ans[0][0]+RETURN2[64:]
        output=True
        Input=False
''' input is the set of initial symptoms seperated by commas and without 
any trailing or leading spaces
'''
def predict(): 
    a=In           
    comp(a)
'''diseases predictor when given symptoms ends here'''

'''skin disease prediction when image is given'''
'''making the one time model '''
def make_model():
	paths=os.path.join("\\".join(os.path.abspath(__file__).split("\\")[:-1]),"dataset")
	paths=os.path.join(paths,"train")
	print(os.getcwd())
	print(os.path.isdir(paths))
	labels=os.listdir(paths)
	print("Found %d Labels" %len(labels))

	'''LOading the features and the labels of the diffrent images '''
	x=[]
	y=[]
	for label in labels:
		folder=os.path.join(paths,label)
		images=os.listdir(folder)
		for img in images:
			loc=os.path.join(folder,img)
			array=cv2.imread(loc)
			x.append(cv2.resize(array,(100,100)).flatten())   
			y.append(label)

	print("Read completed")
	print("size: ",len(x),len(y))
	x=np.asarray(x)
	y=np.asarray(y)
	Y=y.reshape(-1,1)

	from sklearn.preprocessing import OneHotEncoder 

	onecoder=OneHotEncoder(categories='auto')
	Y=onecoder.fit_transform(Y).toarray()

	from sklearn.model_selection import train_test_split

	x_train,x_test,y_train,y_test=train_test_split(x,Y, test_size=0.2,random_state=9)
	y_test[0]

	print('TRAINING\n\n')
	y_test=np.delete(y_test,0,axis=1)
	y_train=np.delete(y_train,0,axis=1)

	print(x_train.shape,y_train.shape)
		
	from sklearn.neighbors import KNeighborsClassifier
	model=KNeighborsClassifier(n_neighbors=3)

	model.fit(x_train,y_train)
	print("TRAINED and saving..\n\n")
	#import pickle
	#filename = 'Knn.pickle'
	#pickle.dump(model, open(filename, 'wb'))
	print("Model NOT saved DUE TO INSUFFICIENT MEMORY(256M)")
	return model

MODEL=""
try:
	MODEL=pickle.load(open(os.path.join("\\".join(os.path.abspath(__file__).split("\\")[:-1]),"Knn.pickle"),"rb"))
except:
	print("Knn Model <Not Found>\nBUILDING MODEL")
	MODEL=make_model()
	#MODEL=pickle.load(open(os.path.join("\\".join(os.path.abspath(__file__).split("\\")[:-1]),"Knn.pickle"),"rb"))
	print("\n\n[done]\n")
	
labels_symptoms=os.path.join("\\".join(os.path.abspath(__file__).split("\\")[:-1]),"dataset")
labels_symptoms=os.path.join(labels_symptoms,"symptoms.txt")

'''solving the collision between diffrent diseases depending on the symptoms user is going through'''
def solve_overlapp(ans):
    global IMG,Out,output,Input,In,NUM,RETURN
    print("solving")
    print(ans)
    with open(labels_symptoms,"r") as f:
        line_no=0
        for line in ans:
            temp=line-line_no-1
            print(temp,line,line_no)
            line_no=line
            while temp:
                f.readline()
                temp-=1
                print("here")
            line=f.readline().split(":")
            symp=line[1].split(",")
            for i in symp:
		R=RETURN[:20]+'d'*NUM+'"'+RETURN[21:]
                Out=R+"do u experience:-"+i+RETURN2[:63]+i+RETURN2[64:]
                NUM+=1
                output=True
                Input=False
                ''' inpuut must be yes or no'''
                print("stuck",Input)
                while not Input:pass
                a=In
                #a=input("do u experience %s " %i)######
                if a=="yes":
                    IMG=0
                    R=RETURN[:20]+'d'*NUM+'"'+RETURN[21:]
                    NUM+=1
                    Out=R+"you may be suffering from:- " +line[0]+RETURN2[:63]+line[0]+RETURN2[64:]
                    output=True
                    Input=False
                    print("you may be suffering from",line[0])####
                    return
    IMG=0
    R=RETURN[:20]+'d'*NUM+'"'+RETURN[21:]
    NUM+=1
    Out=R+"try with a zoomed image of infected area. "
    output=True
    Input=False
    print("try with a zoomed image of infected area. ")######
    
def predict_image(x_pred):
    global IMG,Out,output,Input,NUM,RETURN
    print("in predict")
    import pickle
    model=MODEL
    predicted_values=model.predict_proba(x_pred)
    prob=[]
    found=0
    for i in range(len(predicted_values)):
        if predicted_values[i][0][1] >= 0.2:
            if predicted_values[i][0][1] >= 0.5:
                found=1
            prob.append(i+2)
    if not found:
        prob.insert(0,1)
    if len(prob)>1:
        solve_overlapp(prob)
    else:
        IMG=0
        R=RETURN[:20]+'d'*NUM+'"'+RETURN[21:]
        NUM+=1
        Out=R+"you may be suffering from:- " +"Acne"+RETURN2[:63]+"Acne"+RETURN2[64:]
        output=True
        Input=False
        print("Acne")                            #########
    
def predictor(loc):
    print("in Predictor")
    print(loc)
    array=cv2.imread(loc)
    x_pred=cv2.resize(array,(100,100)).flatten().reshape(1,-1)
    os.remove(loc)
    ans=predict_image(x_pred)
'''skin disease prediction function ends here'''

'''THE FLASK SERVER CODE BEGINS FROM HERE '''

app=Flask(__name__)
PORT = int(os.getenv('PORT', 8000))
'''the url to communicate between the client and the sever using the texts sent by client and out put generated through this url'''
@app.route('/background_process')
def background_process():
    try:
        global Input,In,output,Out,IMG,NUM,RETURN   
        lang=request.args.get('proglang')
        lang=str(lang).lower()
        print("lang=",lang)
        output=False
        arg=0
        if lang not in ["yes","no"]:
        	arg=1
        t=threading.Thread(target=predict,name="thread")
        Input=True
        In=lang
        output=False
        if arg and IMG==0:
        	print("thread began")
        	t.start()

        while not output:pass		# wait until the thread produces response in the Out variable 
        return jsonify(Out)

    except Exception as e:
        print(e)
        return 

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
'''to handle the upload image request from client and to start the image processing thread'''
@app.route('/upload',methods=['GET','POST'])
def upload():
    global IMG,Input,In,output,Out
    print("received")
    file=request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(type(filename),filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print("saved")
        
        image_path=os.path.join("\\".join(os.path.abspath(__file__).split("\\")[:-1]),"uploads")
        image_path=os.path.join(image_path,filename)
        IMG=1

        t_img=threading.Thread(target=predictor,name="thread2",args=(image_path,))
        output=False
        Out=""
        t_img.start()

        while not output:pass
        print(Out)
        return jsonify(Out)
    return jsonify('Image upload faild')
    

@app.route('/',methods=["GET","POST"])
def page():
    try:
        return render_template("upload.html")
    except Exception as e:
        return str(e)

app.run(host='0.0.0.0',debug=True,port=PORT)
