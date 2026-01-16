from flask import Flask,render_template,request
from src.exception import CustomException
from src.logger import logging
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline
import numpy as np
import pandas as pd

application=Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_data():
    if request.method=='GET':
        return render_template('home.html')
    else:
        try:
            # Validate that all fields are provided
            gender = request.form.get('gender')
            race_ethnicity = request.form.get('race_ethnicity')
            parental_level_of_education = request.form.get('parental_level_of_education')
            lunch = request.form.get('lunch')
            test_preparation_course = request.form.get('test_preparation_course')
            reading_score = request.form.get('reading_score')
            writing_score = request.form.get('writing_score')
            
            if not all([gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score]):
                return render_template('home.html', results='Error: All fields are required!')
            
            data=CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=float(reading_score),
                writing_score=float(writing_score)
            )
            pred_df=data.get_data_as_DataFrame()
            print(pred_df)

            predict_pipeline=PredictPipeline()
            results=predict_pipeline.predict(pred_df)
            return render_template('home.html',results=results[0])
        except Exception as e:
            return render_template('home.html', results=f'Error: {str(e)}')
    
if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)