from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory

application= Flask(__name__)
application.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@application.route('/')
def index():
 return render_template('index.html')















