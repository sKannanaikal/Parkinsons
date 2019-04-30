#importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request


app=Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return render_template('predictor.html')


def Predictor(to_predict_list):
    print(to_predict_list)
    to_predict = np.array(to_predict_list) .reshape(1,17)  
    udprs_predictor_model = pickle.load(open("svm_model.pk","rb"))
    result = udprs_predictor_model.predict(to_predict)
    result = np.round(result, 0)
    return result[0]

def Diagnoser(to_predict_list):
    print(to_predict_list)
    to_predict = np.array(to_predict_list) .reshape(1,17)  
    udprs_predictor_model = pickle.load(open("svm_model.pk","rb"))
    
    result = udprs_predictor_model.predict(to_predict)
    diagnosis = ""
    if result < 120:

        if result < 108: 

            if result < 96:

                if result < 84:

                    if result < 72:

                        if result < 60:

                            if result < 48:

                                if result < 36:

                                    if result < 24:

                                        if result < 12:

                                            if result >= 0:

                                                diagnosis = "At this point you are in no point restricted by your disorder individuals will be able to to do all chores without slowness, difficulty or impairment.  Essentially normal. Unaware of any difficulty."
                                        
                                        elif result >= 12: 

                                            diagnosis = "Completely independent. Able to do all chores with some degree of slowness, difficulty and impairment. Might take twice as long.  Beginning to be aware of difficulty."

                                    elif result >= 24:

                                        diagnosis = "Completely independent in most chores. Takes twice as long. Conscious of difficulty and slowness."

                                elif result >= 36:

                                    diagnosis = "Not completely independent. More difficulty with some chores. Three to four times as long in some. Must spend a large part of the day with chores."

                            elif result >= 48:

                                diagnosis = " Some dependency. Can do most chores, but exceedingly slowly and with much effort. Errors; some impossible."

                        elif result >= 60:

                            diagnosis ="More dependent. Help with half, slower, etc. Difficulty with everything."

                    elif result >= 72:

                        diagnosis = "Very dependent. Can assist with all chores, but few alone."

                elif result >= 84:

                        diagnosis = "With effort, now and then does a few chores alone or begins alone. Much help needed."        

            elif result >= 96:

                diagnosis = 'Completely depdent can be a slight help with some chores constantly around someone'

        elif result >= 108:

            diagnosis = 'Patients are completely dependent on others however can still somewhat perform tasks'

    elif result >= 120:

        diagnosis = "Dangerous point patients will have Vegetative functions such as swallowing, bladder and bowel functions are not functioning. Bedridden."

    return diagnosis

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        print('dict collection', end='\n')
        print(to_predict_list)
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = Predictor(to_predict_list)
        diagnosis = Diagnoser(to_predict_list)
        return render_template("diagnosis.html",prediction=result, diagnosis=diagnosis)

if __name__ == '__main__':
	app.run(debug=True)
