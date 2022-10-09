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

def folder():
    st.sidebar.markdown('___')
    st.sidebar.markdown('## BCCD')
    st.sidebar.markdown('___')
    model_type = st.sidebar.radio('Modell', ['BCCD', 'Raabin', 'LISC'])
    st.image('data/BCCD/neut_45_5733.jpeg', use_column_width=True, caption='Beispiel von BCCD')
    if st.button('Start'):
        folder_performance('BCCD', 'BCCD',  model_path=f'data/{model_type}.pkl',train_path=f'data/BCCD/x_train.npy')
