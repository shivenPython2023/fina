from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session





app= Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/')
def index():
 return 'helo world',render_template('index.html')






UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}








app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER





def allowed_file(filename):
 return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS






@app.route('/color.html', methods=['GET', 'POST'])
@app.route('/color', methods=['GET', 'POST'])
def color_html():
   if request.method == 'POST':
       if 'file' not in request.files:
           return redirect(request.url)
       file = request.files['file']
       if file.filename == '':
           return redirect(request.url)
       if file and allowed_file(file.filename):
           filename = secure_filename(file.filename)
           file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
           input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)


           file_str = str(filename)
           file_name = remove_extension(file_str)
           session['file_name']=file_name
           # Process images
           # Simplify the input image
           simplified_image_paths = {}
           for i in range(2, 11):
               simplified_image_paths[i] = os.path.join(app.config['UPLOAD_FOLDER'], file_name + f'simplified{i}.jpg')
               simplify_image(input_image_path, simplified_image_paths[i], i)


           simplified_image_paths_for1=os.path.join(app.config['UPLOAD_FOLDER'], file_name + f'simplified{"2"}.jpg')
           simplify_image_for1(input_image_path, simplified_image_paths_for1, 2)


           outline_image_paths = {}
           for i in range(2, 6):
               outline_image_paths[i] = os.path.join(app.config['UPLOAD_FOLDER'], file_name + f'outline_{i}.jpg')
               create_outline(simplified_image_paths[i], outline_image_paths[i])
          
           grid_image_paths={}
           for i in range(2, 6):
               grid_image_paths[i] = os.path.join(app.config['UPLOAD_FOLDER'], file_name + f'grid{i}.jpg')
               create_grid(outline_image_paths[i], grid_image_paths[i])


           color_image_paths={}
           for i in range(2, 11):
               color_image_paths[i] = os.path.join(app.config['UPLOAD_FOLDER'], file_name + f'color{i}.jpg')
               get_dominant_colors(simplified_image_paths[i], color_image_paths[i], i)


           output_image_path_for_canvas = os.path.join(app.config['UPLOAD_FOLDER'], file_name + 'plain.jpg')
           create_white_canvas(input_image_path, output_image_path_for_canvas)




           # Render sketch.html with processed images
           grid2= '/uploads/'+file_name+ 'grid2.jpg'
           grid3= '/uploads/'+file_name+ 'grid3.jpg'
           grid4= '/uploads/'+file_name+ 'grid4.jpg'
           return render_template('sketch.html', plain=file_name + 'plain.jpg', sketch2=file_name + 'outline_2.jpg', sketch3=file_name + 'outline_3.jpg', sketch4=file_name + 'outline_4.jpg', sketch5=file_name + 'outline_5.jpg', filename=file_name, grid_2= grid2, grid_3= grid3, grid_4= grid4,)
   return render_template('color.html')






@app.route('/realcolor.html', methods=['GET', 'POST'])
def realcolor_html():
   file_name = session.get('file_name')  # Get file_name from session
   if not file_name:
       # Handle the case when file_name is not found in session
       return "Error: File name not found in session"


   # Use file_name to construct image paths
   return render_template('realcolor.html', im2=file_name + 'simplified2.jpg', im3=file_name + 'simplified3.jpg', im4=file_name + 'simplified4.jpg', im5=file_name + 'simplified5.jpg', im6=file_name + 'simplified6.jpg', im7=file_name + 'simplified7.jpg', im8=file_name + 'simplified8.jpg', im9=file_name + 'simplified9.jpg', im10=file_name + 'simplified10.jpg',
                          col2= file_name + 'color2.jpg', col3= file_name + 'color3.jpg', col4= file_name + 'color4.jpg', col5= file_name + 'color5.jpg', col6= file_name + 'color6.jpg', col7= file_name + 'color7.jpg', col8= file_name + 'color8.jpg',  col9= file_name + 'color9.jpg', col10= file_name + 'color10.jpg',)




@app.route('/uploads/<filename>')
def uploaded_file(filename):
 return send_from_directory(app.config['UPLOAD_FOLDER'], filename)








if __name__ == '__main__':
 app.run()

















