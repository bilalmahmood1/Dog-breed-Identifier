#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make a simple web page that predicts dog breed
@author: bilal
"""


from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
import requests
import pandas as pd
from bokeh.palettes import Reds5
from bokeh.plotting import figure
from bokeh.embed import components
import os

KERAS_REST_API_URL = "35.200.244.216/predict"
IMAGE_PATH = os.path.join("/var/www/html/Dog-breed-Identifier","static/img/")


app = Flask(__name__)


photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = IMAGE_PATH
configure_uploads(app, photos)

@app.route('/', methods=['GET'])
def home(error = ""): 
    photo_name = 'dog_picture.jpg'
    predictions = make_predictions_api(photo_name)
    if predictions is not None:
        top_class = get_top_class(predictions)
        script,div = make_predictions_visual(predictions)
    else:
        predictions = []
        top_class = None
        script,div = None, None
    
    
    return render_template('upload_form.html', 
                           photo_name = photo_name,
                           predictions = predictions,
                           prediction_result = "prediction.png",
                           top_class = top_class,
                           script = script,
                           div = div,
                           error = error)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        """Make predict and returns the result with the visual embedded"""
        if request.method == 'POST' and 'photo' in request.files:
            filename = photos.save(request.files['photo'])
            predictions = make_predictions_api(filename)
            if predictions is not None:
                top_class = get_top_class(predictions)
                script,div = make_predictions_visual(predictions)
            else:
                predictions = []
                top_class = None
                script,div = None, None
            return render_template('upload_form.html',
                                   photo_name = filename,
                                   predictions = predictions,
                                   prediction_result = "prediction.png",
                                   top_class = top_class,
                                   script = script,
                                   div = div)
        else:
            return home("Please upload image of your dog")
    except Exception as error:
        print(error)
        return home("Please upload jpeg image of your dog")
       
    
    

def make_predictions_visual(predictions):
    """returns the html elements of the visual generated"""
    df = pd.DataFrame(predictions)    
    
    df.sort_values(by="probability",
                   ascending = False,
                   inplace = True)
    
    labels = df['label'].values
    sizes = df['probability'].values
          
    p = figure(x_range = labels, plot_height = 350,
               title = "Confidence over dog breeds",
               tools = "", toolbar_location = None)
    p.vbar(x=labels, top=sizes, width=0.9, color=Reds5)
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.y_range.end = 1
    
    p.xaxis.axis_label = 'Potential Breeds'
    p.yaxis.axis_label = 'Probability'
    
    script, div = components(p)
    return script,div
    

def get_top_class(predictions):
    """Return top class, i.e. with the highest probability"""
    df = pd.DataFrame(predictions)    
    
    df.sort_values(by="probability",
                   ascending = False,
                   inplace = True)
    return df['label'].iloc[0]


def make_predictions_api(filename):
    """Make predicts on each uploaded file by making an API call to the 
    server with the deep learning model loaded"""
    # initialize the Keras REST API endpoint URL along with the input
    # image path

    image_path = IMAGE_PATH + filename
    image_path = os.path.join(os.getcwd(), image_path)
    try:
        # load the input image and construct the payload for the request
        image = open(image_path, "rb").read()
        payload = {"image": image}
        # submit the API request
        r = requests.post(KERAS_REST_API_URL, files=payload).json()
        if r["success"]:
            result = r['predictions']
    
    except Exception as error:
        print(error)
        result = None
    
    finally:
        return result
    
    
if __name__ == '__main__':
    app.run(port = 80,
            debug=True)
