from flask import Flask, render_template 
application=Flask(__name__)

@application.route('/')
def hello_world():
    return render_template('index.html')
