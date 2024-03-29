from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory

application= Flask(__name__)
application.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt
import matplotlib
import warnings
from sklearn.cluster import KMeans
import webcolors
from werkzeug.utils import secure_filename

@application.route('/')
def index():
    return render_template('index.html')














