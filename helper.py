from cgi import test
import re
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import io
from model import *
import os
import glob
import cv2
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from Functions import segmentation, feature_extractor

def svmpredict(img_path, model='images/svm/t.pkl', x_train_path='images/svm/x_train.npy', up=False):
    model = joblib.load(model)
    if up == False: img = cv2.imread(img_path)
    else: img = img_path
    x_train = np.load(x_train_path)

    print(feature_extractor(img=img, min_area=100))
    ncl_detect, error, ftrs = feature_extractor(img=img, min_area=100)
    if ncl_detect:
        ftrs = np.array(ftrs).reshape(1, -1)
        # normalize feature using max-min way
        mn, mx = x_train.min(axis=0), x_train.max(axis=0)
        ftrs = (ftrs - mn)/(mx - mn)
        print(ftrs)
        pred = model.predict(ftrs)
        return pred[0]
    else:
        return error

def load_image(image_file, svm=False):
    img = Image.open(image_file)
    # resize image to 2592x1944
    if not svm:
        
        image = img.resize((2592, 1944))
        # 24 bit color depth
        image = image.convert('RGB')
        image.save("images/tmp.jpg")
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        return image
    else: 
        img = img.convert('RGB')
        open_cv_image = np.array(img)
        st.image(open_cv_image, use_column_width=True)
        # open_cv_image = open_cv_image[:, :, ::-1].copy() 
        #cv2.imwrite('images/svm/tmp.jpg', open_cv_image)
        return open_cv_image
        # save image
    
def merge_image():
    images = [Image.open(x) for x in ['images/svm/nuc.jpg', 'images/svm/ROC.jpg']]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    st.image(new_im, caption='Links Nukleus Rechts Konvexe Hülle', use_column_width=True)

def remove_artifacts(img):
    image = cv2.imread(img)
    # crop 100 px from left and top
    image = image[100:, 100:]
    # crop 50 px bottom
    image = image[:-50, :]
    # crop 110 px right
    image = image[:, :-110]
    # overwrite image
    cv2.imwrite(img, image)

def read(prediction, test=False):
    if prediction == 1:
        prediction = 'Neutrophil'
    elif prediction == 2:
        prediction = 'Lymphozyt'
    elif prediction == 3:
        prediction = 'Monozyt'
    elif prediction == 4:
        prediction = 'Eosinophil'
    elif prediction == 5:
        prediction = 'Basophil'
    else:
        prediction = 'Keine Blutzelle gefunden'
    if test:
        return prediction
    else:
        st.write(f'Ergebnis: {prediction}')

def large_img_det(img, image_name='tmp'):
    image = cv2.imread(img)
    # Iterate over the image in 600x600 pixel steps, crop out each 600x600 pixel image and predict it
    # save the predictions in a list
    predictions = []
    for i in range(0, image.shape[0], 600):
        for j in range(0, image.shape[1], 600):
            # crop image
            crop_img = image[i:i+600, j:j+600]
            # save image
            cv2.imwrite(f'images/svm/{image_name}.jpg', crop_img)
            # predict
            predictions.append(read(svmpredict(img_path=f'images/svm/{image_name}.jpg'), test=True))
    # return the most common prediction
    common = max(set(predictions), key=predictions.count)
    st.write(f'Ergebnis: {common}')
    st.write(predictions)
def folder_performance(folder, name, model_path, train_path):
    # predict every image in folder
    predictions = []
    st.write(f'Performance für {name}')

    if name == 'BCCD':
        for img in glob.glob(f'data/{folder}/*.jpeg'):
            # st.write('hello world')
            predictions.append(svmpredict(img_path=img, model=model_path, x_train_path=train_path))
    elif name == 'LISC':
        for img in glob.glob(f'data/{folder}/*.bmp'):
            predictions.append(svmpredict(img_path=img))
    #compare predictions with ground truth, thats in the folder named test.json
    with open(f'data/{folder}/Test.json') as f:
        data = json.load(f)
    # the json is a dictionary with the image name as key and the ground truth as value
    # create a list with the ground truth

    ground_truth = []
    for key, value in data.items():
        ground_truth.append(value)
        # show ground truth in a table
    df = pd.DataFrame({'Bild': list(data.keys()), 'Ergebnis': ground_truth})
    #st.table(df)
    # compare predictions with ground truth

    # create a bar chart with the predictions
    # create a list with the unique values
    unique = list(set(predictions))
    # create a list with the counts of the unique values
    counts = []
    for i in unique:
        counts.append(predictions.count(i))
    # create a list with the counts of the ground truth
    gt_counts = []
    for i in unique:
        gt_counts.append(ground_truth.count(i))
    # create a list with the names of the unique values
    names = []
    for i in unique:
        if i == 1:
            names.append('Neutrophil')
        elif i == 2:
            names.append('Lymphozyt')
        elif i == 3:
            names.append('Monozyt')
        elif i == 4:
            names.append('Eosinophil')
        elif i == 5:
            names.append('Basophil')
        else:
            names.append('Keine Blutzelle gefunden')
    # create a list with the names of the ground truth
    gt_names = []
    for i in unique:
        if i == 1:
            gt_names.append('Neutrophil')
        elif i == 2:
            gt_names.append('Lymphozyt')
        elif i == 3:
            gt_names.append('Monozyt')
        elif i == 4:
            gt_names.append('Eosinophil')
        elif i == 5:
            gt_names.append('Basophil')
        else:
            gt_names.append('Keine Blutzelle gefunden')
    # create a dataframe with the names, counts and ground truth counts
    df = pd.DataFrame({'names': names, 'counts': counts, 'ground_truth': gt_counts})

    # create a bar chart with the predictions and the ground truth add both to the same chart and legend
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['names'], y=df['counts'], name='Predictions', marker_color='#50fa7b'))
    fig.add_trace(go.Bar(x=df['names'], y=df['ground_truth'], name='Ground Truth', marker_color='#bd93f9'))

    fig.update_layout(barmode='group')
    # change containenr width to columnn width
    st.plotly_chart(fig, use_container_width=True)
    # display dataframe 
    st.dataframe(df, use_container_width=True)