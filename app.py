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

from fold import folder
from home import home
from wbc import wbc
# ---
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

if __name__ == '__main__':
    with st.sidebar:
        selection = option_menu(
            None,
            options=['Home', 'WBC', 'WBC-Performance'],
            icons=['house', 'graph-up','file-earmark-bar-graph'],
            styles={
                'icon': {'font-size': '18px'},
                'nav-link-selected': {'background-color': '#3e0e22'}
            },
            orientation='vertical',
        )

    if selection == 'Home':
        home()
    elif selection == 'WBC':
        wbc() 
    elif selection == 'WBC-Performance':
        folder()
