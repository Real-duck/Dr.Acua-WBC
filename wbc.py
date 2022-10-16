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
from Functions import segmentation, feature_extractor

from helper import *
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def wbc():
    st.sidebar.markdown('___')
    option_wbc = st.sidebar.radio('Optionen', ['Segmentieren', 'Erkennen'])
    image = "Bild2"
    
    if st.selectbox('Upload oder Testbild?', ['Testbild', 'Upload']) == 'Upload':
        image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
        st.warning('Wenn man Bilder in einem benutzbaren Format haben will, kann man Bilder aus dem LISC oder BCCD Datensatz nehmen.') 
        if image_file is not None:
            load_image(image_file, svm=True)
            # To See details
            file_details = {"filename":image_file.name, "filetype":image_file.type,
                            "filesize":image_file.size}
            st.write(file_details)
            image = "tmp"
            st.image(Image.open(f'images/svm/{image}.jpg'), caption='Bild', use_column_width=True)
    else:
        st.markdown('___')
        st.markdown('# Bild auswählen:')
        image = st.radio('', ['Bild1', 'Bild2', 'Bild3', 'Bild4', 'Bild5'])
        st.image(Image.open('images/svm/' + image + '.jpg'), caption='Testbild')
    start = st.button('Start')
    shape = cv2.imread(f'images/svm/{image}.jpg').shape
    if option_wbc == 'Segmentieren' and start:
        
        segmentation(cv2.imread(f'images/svm/{image}.jpg'))
        # if image is large don't merge
        if shape[0] > 800 or shape[1] > 800:
            st.image(Image.open('images/svm/nuc.jpg'), caption='Segmentiertes Bild', use_column_width=True)
        else:
            merge_image()
    elif option_wbc == 'Erkennen' and start:

        if shape[0] < 800 or shape[1] < 800:
            prediction =svmpredict(img_path=f'images/svm/{image}.jpg')
            read(prediction)
        else:
            st.warning('Die benutzte Methode ist aus zeitlichen gründen nicht ausgereift.')
            st.write(large_img_det(img=f'images/svm/{image}.jpg', image_name=image))
