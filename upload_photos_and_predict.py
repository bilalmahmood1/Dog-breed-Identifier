#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make a simple web page that predicts dog breed
@author: bilal
"""


from flask import Flask, render_template, request
from flask.ext.uploads import UploadSet, configure_uploads, IMAGES
import requests
import pandas as pd
from bokeh.palettes import Reds5
from bokeh.plotting import figure
from bokeh.embed import components

KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "static/img/"


app = Flask(__name__)

photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = IMAGE_PATH
configure_uploads(app, photos)

@app.route('/', methods=['GET'])
def home(): 
    photo_name = 'dog_picture.jpg'
    predictions = make_predictions_api(photo_name)
    top_class = get_top_class(predictions)
    script,div = make_predictions_visual(predictions)
    return render_template('upload_form.html', 
                           photo_name = photo_name,
                           predictions = predictions,
                           prediction_result = "prediction.png",
                           top_class = top_class,
                           script = script,
                           div = div)


@app.route('/predict', methods=['POST'])
def predict():
    """Make predict and returns the result with the visual embedded"""
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        predictions = make_predictions_api(filename)
        top_class = get_top_class(predictions)
        script,div = make_predictions_visual(predictions)
        return render_template('upload_form.html',
                               photo_name = filename,
                               predictions = predictions,
                               prediction_result = "prediction.png",
                               top_class = top_class,
                               script = script,
                               div = div)
    
    

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

    # load the input image and construct the payload for the request
    image = open(image_path, "rb").read()
    payload = {"image": image}
    
    # submit the request
    r = requests.post(KERAS_REST_API_URL, files=payload).json()
    
    result = ""
    # ensure the request was successful
    if r["success"]:
        result = r['predictions']
    # otherwise, the request failed
    else:
        result = None
        
    return result
    
    
if __name__ == '__main__':
    app.run(port = 5001,
            debug=True)