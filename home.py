import streamlit as st
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

# ---
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def home():
    st.warning('Weil Streamlit die Websie verlangsamt, ist das Plättchen-Modul gesperrt worden.') 
    st.markdown('## Was ist Dr Acula?')
    st.markdown('Inspiriert vom berüchtigten Vampir - mit seiner eigenen Netflix Serie - Dracula, hat sich '
               'der Vampir Dr. Acula dazu entschieden sein Wissen über Blut zu veröffentlichen. Er kann '
               'eine sich oft wiederholende, aber auch wichtige Aufgabe, vom Zählen der Arten von Blutkörperchen übernehmen.')
    st.markdown('___')

    st.markdown('## Wie benutze ich die Website?')
    st.markdown('Auf dieser Seite gibt es verschiedene Module, das CBC Modul kann Blutzellen von einem Blutbild, welches mit einem '
                'Mikroskop aufgenommen wurde, segmentieren und typisieren. Das WBC Modul kann weiße Blutzellen Klassifizieren. '
                'Dafür muss man aktuell die weiße Blutzelle ausschneiden und dann hochladen. Bei größeren Bildern wird das hochgeladene '
                'Bild aktuell nur in 600x600 aufgeteilt. In der Zukunft soll dafür eine KI benutzt werden, welche alle Zellen automatisch '
                'findet und ausschneided.')
    st.image(Image.open('images/reprs.jpg'), caption='Dimensionen sind falsch', width=400)
    st.markdown('___')
    #st.markdown('Github: Kommt Bald!')

    #st.markdown('## Wie funktioniert die Website? ')
    #           'Dies ist eine sehr komprimierte Version von Dr. Acula. ')

    #st.markdown('## Die Einstellungen ')

    #st.markdown('#### Modeltypen')
    #st.markdown(
    #            'SegNet und U-Net sind die beiden Architekturen, die von Dr. Acula verwendet werden kann. Da beide die Farben der einzelnen Körperchen ' 
    #            '')

    #st.markdown('#### Transformationen')
    #st.markdown(
    #    '*Mit diesen Methoden kann das Ergebnis leicht überprüft werden. Der Datensatz kann mit den Ergebnissen auch erweitert werden.*')

    #st.markdown('___')
    #st.markdown('___Threshold___:')
    #st.markdown('Beim Thresholding werden Vordergrundpixel vom Hintergrund seperiert. '
    #            ' Die Pixel, die zum Vordergrund gehören sind Weiß, die, die zum Hintergrund gehören sind Schwarz. [Mehr zum benutzten Verfahren](https://www.wikiwand.com/en/Otsu%27s_method)')
    # st.image(Image.open('images/threshold_edge_mask.png'), caption='Thresholding für rote Blutkörperchen',
    #             width=400)
    # st.markdown('___')# 

    # st.markdown('___Hough Transform___:')
    # st.markdown(
    #     'Diese Methode kann spezifische Formen erkennen. Da Blutkörperchen oft kreisförmig sind, wird hier Circle Hough Transform verwendet. '
    #     'Das Ergebnis kann für das Zählen einer Blutkörperchen Art benutzt werden. [Mehr zum benutzten Verfahren](https://sbme-tutorials.github.io/2021/cv/notes/4_week4.html)')
    # st.image(Image.open('images/hough_transform.png'), caption='Hough Transform für rote Blutkörperchen',
    #             width=400)
    # st.markdown('___')

    # st.markdown('___Component labeling___:')
    # st.markdown(
    #     'Dieser Algorithmus erkennt verbundene Objekte in einem binären Bild, also das Ergebnis der Segmentation.'
    #     'Component labeling kann auch für das Zählen benutzt werden. [Mehr zum benutzten Verfahren] (https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/)')
    # st.image(Image.open('images/component_labeling.png'), caption='Component labeling für rote Blutkörperchen',
    #             width=400)
    # st.markdown('___')

    # st.markdown('___Distance transform___:')
    # st.markdown(
    #     'Diese Methode nimmt ein binäres Bild und gibt ein grayscale Bild zurück, das die Distanz zwischen den Pixeln repräsentiert. '
    #     'Je weiter ein Pixel vom Rand entfernt ist, desto heller wird er sein. [Mehr zum benutzten Verfahren](https://homepages.inf.ed.ac.uk/rbf/HIPR2/distance.htm)')
    # st.image(Image.open('images/distance_transform.png'), caption='Distance transform für rote Blutkörperchen',
    #             width=400)

    st.markdown('## Wie funktioniert die Typisierung?')
    st.markdown('Es wird eine Support vector machine für die Typisierung benutzt. Vereinfacht funktioniert diese mithilfe einer '
                'Trennlinie, welche die Datenpunkte trennt. Die Datenpunkte, die auf der einen Seite der Linie liegen, werden als '
                'eine Klasse und die anderen als eine andere Klasse betrachtet. Die Trennlinie wird so gewählt, dass sie die '
                'Datenpunkte mit einem so groß wie möglichen Abstand trennt. Die Trennlinie ist als Kernel bekannt, der Name kommt von dem Kernel Trick.')
    st.markdown('GitHub: <https://github.com/DavidRutkevich/DR.Acula-SVM>')
    st.markdown('___')
    st.markdown('## Wie werden die Datenpunkte ermittelt? ')
    st.markdown('Die Datenpunkte werden mit Hilfe von Segmentierung und Feature Extraction ermittelt. '
                'Die Segmentierung basiert auf einen neuen Algorithmus, welcher die Nuklei schnell und genau segmentiert. '
                'Um das Zytoplasma zu extrahieren, wird der Nukleus Segmentiert und dann die Konevexe Hülle von diesem ermittelt. '
                'Das Zytoplasma in dieser Hülle reicht voll kommen aus, um wichtige Eigenschaften zu extrahieren.')
    st.markdown('___')

    st.markdown('### Blutzellen was das??')
    st.image(Image.open('images/re1dark.png'))

    
    st.markdown('#### Transformationen')
    st.markdown(
        '*Mit diesen Methoden kann das Ergebnis leicht überprüft werden. Der Datensatz kann mit den Ergebnissen auch erweitert werden.*')
    st.markdown('___')
    st.markdown('___Threshold___:')
    st.markdown('Beim Thresholding werden Vordergrundpixel vom Hintergrund seperiert. '
                ' Die Pixel, die zum Vordergrund gehören sind Weiß, die, die zum Hintergrund gehören sind Schwarz. [Mehr zum benutzten Verfahren](https://www.wikiwand.com/en/Otsu%27s_method)')
    st.image(Image.open('images/threshold_mask.png'), caption='Thresholding für rote Blutkörperchen',
                width=400)
    st.markdown('___')# 
    st.markdown('___Hough Transform___:')
    st.markdown(
        'Diese Methode kann spezifische Formen erkennen. Da Blutkörperchen oft kreisförmig sind, wird hier Circle Hough Transform verwendet. '
        'Das Ergebnis kann für das Zählen einer Blutkörperchen Art benutzt werden. [Mehr zum benutzten Verfahren](https://sbme-tutorials.github.io/2021/cv/notes/4_week4.html)')
    st.image(Image.open('images/hough_transform.png'), caption='Hough Transform für rote Blutkörperchen',
                width=400)


    st.markdown('___')
    st.markdown('___Distance transform___:')
    st.markdown(
        'Diese Methode nimmt ein binäres Bild und gibt ein grayscale Bild zurück, das die Distanz zwischen den Pixeln repräsentiert. '
        'Je weiter ein Pixel vom Rand entfernt ist, desto heller wird er sein. Dieses Verfahren kann man aktuell nur isoliert unter der Option "Zählen?" benutzen[Mehr zum benutzten Verfahren](https://homepages.inf.ed.ac.uk/rbf/HIPR2/distance.htm)')
    st.image(Image.open('images/output.png'), caption='Distance transform für rote Blutkörperchen',
                 width=400)
