#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Combined App using uploading and prediction @author: bilal """


# import the necessary packages
from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
import pandas as pd
from bokeh.palettes import Reds5
from bokeh.plotting import figure
from bokeh.embed import components
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import os


IMAGE_PATH = os.path.join("/var/www/html/Dog-breed-Identifier","static/img/")

#model = None

global model
model = ResNet50(weights="imagenet")

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



def load_model():
    """load the pre-trained Keras model (here we are using a model"""
    global model
    model = ResNet50(weights="imagenet")
        
def prepare_image(image, target):
    """Preparing image for inputting into Neural network"""
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

def make_predictions_api(filename):
    """Make predictions on the uploaded image"""
    data = None
    try:
        ## loading uploaded image
        image_path = IMAGE_PATH + filename
        image = Image.open(open(image_path,'rb'))
    
        # preprocess the image and prepare it for classification
        image = prepare_image(image, target=(224, 224))
    
        # classify the input image and then initialize the list
        # of predictions to return to the client
        preds = model.predict(image)
        results = imagenet_utils.decode_predictions(preds)
        data = []
    
        # loop over the results and add them to the list of
        # returned predictions
        for (imagenetID, label, prob) in results[0]:
            r = {"label": label, "probability": float(prob)}
            data.append(r)
    
        return data['predictions']
    
    except Exception as error:
        print(error)
        return data
   
    

#print('Loading pretrained model')
#load_model()
#print('Model Loaded and application is running')
#app.run(port=80, debug=True)    
    
    
