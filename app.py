import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
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
from flask import session, redirect, url_for








app= Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/')
def index():
 return render_template('index.html')






UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}








app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER






def compress_image(image_path, new_image_path, new_size):
    image = Image.open(image_path)
    w, h = image.size
    if h > w:
        new_h = new_size
        new_w = int(w * (new_size / h))
    else:
        new_w = new_size
        new_h = int(h * (new_size / w))
    image = image.resize((new_w, new_h), resample=Image.BICUBIC)
    print("image simplified")
    image.save(new_image_path, optimize=True, quality=60)


def allowed_file(filename):
 return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def simplify_image_for1(input_image_path, output_image_path, num_clusters):
   try:
       if num_clusters is None:
           print("Error: Invalid number of clusters")
           return


       image = cv2.imread(input_image_path)


       # Calculate the kernel size as 1% of the input image size, with a minimum of 3x3 and a maximum of (image.shape[1]-1) x (image.shape[0]-1)
       kernel_size = int(min(image.shape[1]-1, image.shape[0]-1) * 0.01)
       kernel_size = max(3, min(kernel_size, image.shape[1]-1, image.shape[0]-1))


       # Apply Gaussian blur with the calculated kernel size
       ksize = (7, 7)
       smoothed_image = cv2.GaussianBlur(image, ksize, 0)


       # Convert the smoothed image to a 2-dimensional array of 32-bit floating-point numbers
       smoothed_image_flattened = smoothed_image.reshape((-1, 3)).astype(np.float32)


       # Apply k-means clustering to the pixels
       criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
       _, labels, centers = cv2.kmeans(smoothed_image_flattened, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


       # Convert the centers to uint8
       centers = np.uint8(centers)


       # Get the segmented image
       segmented_image = centers[labels.flatten()]


       # Reshape the segmented image to match the original image shape
       segmented_image = segmented_image.reshape(image.shape)


       # Save the segmented image
       cv2.imwrite(output_image_path, segmented_image)


   except Exception as e:
       print(f"Error: {e}")


def simplify_image(input_image_path, output_image_path, num_clusters):
  try:
      # Calculate the number of clusters (dominant colors)
      if num_clusters is None:
          print("Error: Invalid number of clusters")
          return




      # Read the input image again
      image = cv2.imread(input_image_path)




      # Reshape the image to be a list of pixels
      pixels = image.reshape((-1, 3)).astype(np.float32)


       # Apply Gaussian blur with a specified kernel size
      kernel_size = (5, 5)  # Adjust the kernel size as needed
      smoothed_image = cv2.GaussianBlur(pixels, kernel_size, 0)
      # Apply k-means clustering to the pixels
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
      _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)




      # Convert the centers to uint8
      centers = np.uint8(centers)




      # Get the segmented image
      segmented_image = centers[labels.flatten()]




      # Reshape the segmented image to match the original image shape
      segmented_image = segmented_image.reshape(image.shape)




      # Save the segmented image
      cv2.imwrite(output_image_path, segmented_image)




  except Exception as e:
      print(f"Error in simplify_image: {e}")




def create_outline(input_image_path, output_image_path):
   try:
       matplotlib.use('Agg')
       # Loading Original Image
       img = cv2.imread(input_image_path)
       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


       # Create a subplot with the same aspect ratio as the input image
       fig, axs = plt.subplots(1, 1, figsize=(img.shape[1]/100, img.shape[0]/100))


       # Hide axes
       axs.axis('off')


       # Converting Image to GrayScale
       img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


       # Inverting the Image
       img_invert = cv2.bitwise_not(img_gray)


       # Applying bilateral filter for noise reduction
       img_filtered = cv2.bilateralFilter(img_invert, 9, 75, 75)


       # Converting to Pencil Sketch
       final = cv2.divide(img_gray, 255 - img_filtered, scale=255)


       # Add thinner black border to the image
       bordered_image = cv2.copyMakeBorder(final, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))


       # Update the canvas with the final sketch
       axs.imshow(bordered_image, cmap="gray")


       # Save the final sketch image directly without displaying
       plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, transparent=True)
       plt.close(fig)  # Close the figure to avoid displaying it


   except Exception as e:
       print(f"Error in create_outline: {e}")




def create_white_canvas(input_image_path, output_image_path):
  try:
      # Open the input image
      input_image = Image.open(input_image_path)




      # Get the size of the input image
      width, height = input_image.size




      # Create a new white canvas with the same size as the input image
      white_canvas = Image.new("RGB", (width, height), color="white")




      # Add a black border to the white canvas
      final_canvas = cv2.copyMakeBorder(np.array(white_canvas), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))




      # Save the white canvas with a black border as a new image
      Image.fromarray(final_canvas).save(output_image_path)




  except Exception as e:
      print(f"Error: {e}")


def remove_extension(file_path):
   # Find the position of the last dot in the file path
   dot_index = file_path.rfind('.')
  
   # If a dot is found and it's not the first character
   if dot_index != -1 and dot_index > 0:
       # Extract the substring before the last dot
       return file_path[:dot_index]
  
   # If no dot is found or it's the first character, return the original file path
   return file_path


def add_grid2(image, output_image_path):
  # Get image dimensions
  image_width, image_height = image.size




  # Define the minimum and maximum number of rows and columns
  min_rows, min_columns = 3, 3
  max_rows, max_columns = 5, 5




  # Calculate the optimal number of rows and columns based on the image dimensions
  num_rows = max(min_rows, min(int(np.ceil(np.sqrt(image_height))), max_rows))
  num_columns = max(min_columns, min(int(np.ceil(np.sqrt(image_width))), max_columns))




  # Calculate the spacing between rows and columns
  row_spacing = image_height // num_rows
  column_spacing = image_width // num_columns




  # Create a new image with an alpha channel
  grid_image = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
  draw = ImageDraw.Draw(grid_image)




  # Draw horizontal grid lines
  for i in range(1, num_rows):
      y = i * row_spacing
      draw.line([(0, y), (image_width, y)], fill=(255, 0, 0, 255), width=2)




  # Draw vertical grid lines
  for i in range(1, num_columns):
      x = i * column_spacing
      draw.line([(x, 0), (x, image_height)], fill=(255, 0, 0, 255), width=2)




  # Convert the grid image to a numpy array
  grid_array = np.array(grid_image)




  # Convert the original image to a numpy array
  image_array = np.array(image.convert("RGBA"))




  # Blend the two images
  blended_array = (grid_array * (grid_array[:, :, 3] / 255.0)[:, :, None] +
                   image_array * (1.0 - grid_array[:, :, 3] / 255.0)[:, :, None]).astype(np.uint8)




  # Create a new image from the blended array
  blended_image = Image.fromarray(blended_array, 'RGBA')




  # Save the resulting image
  blended_image.save(output_image_path, format="PNG")




def create_grid(input_image_path, output_image_path):
  # Load the input image (contour image from the "Drawing" stage)
  contour_image = Image.open(input_image_path)
   # Convert the Pillow Image to a NumPy array
  contour_array = np.array(contour_image)




  # Apply border
  final_grid = cv2.copyMakeBorder(contour_array, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))




  # Now, add the grid to the contoured image
  add_grid2(Image.fromarray(final_grid), output_image_path)




def plain_grid(image, output_image_path):
  # Read the image using OpenCV
  img = cv2.imread(image)
  image_height, image_width, _ = img.shape




  # Define the minimum and maximum number of rows and columns
  min_rows, min_columns = 5, 5
  max_rows, max_columns = 7, 7




  # Calculate the optimal number of rows and columns based on the image dimensions
  num_rows = max(min_rows, min(math.ceil(math.sqrt(image_height)), max_rows))
  num_columns = max(min_columns, min(math.ceil(math.sqrt(image_width)), max_columns))




  # Calculate the spacing between rows and columns
  row_spacing = image_height // num_rows
  column_spacing = image_width // num_columns




  # Create a new image with an alpha channel
  grid_image = Image.new("RGBA", (image_width, image_height), (255, 255, 255, 0))  # Set background to white
  draw = ImageDraw.Draw(grid_image)




  # Draw horizontal grid lines
  for i in range(1, num_rows):
      y = i * row_spacing
      draw.line([(0, y), (image_width, y)], fill=(255, 0, 0, 255), width=5)




  # Draw vertical grid lines
  for i in range(1, num_columns):
      x = i * column_spacing
      draw.line([(x, 0), (x, image_height)], fill=(255, 0, 0, 255), width=5)




  # Convert the PIL image to OpenCV format
  final = cv2.cvtColor(np.array(grid_image), cv2.COLOR_RGBA2BGR)
   # Add border using OpenCV
  final = cv2.copyMakeBorder(final, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))




  # Save the final image
  cv2.imwrite(output_image_path, final)




def rgb_to_closest_color_name(rgb_tuple):
   try:
       # Find the closest color name
       closest_color = min(webcolors.CSS3_HEX_TO_NAMES.items(),
                           key=lambda item: np.linalg.norm(np.array(rgb_tuple) - np.array(webcolors.hex_to_rgb(item[0]))))
       color_name = closest_color[1]
   except ValueError:
       color_name = "N/A"
   return color_name


def get_dominant_colors(image_path, output_path, num_colors):
   img = Image.open(image_path)
   img = img.resize((100, 100))
   img_array = np.array(img)
   pixels = img_array.reshape((-1, 3))


   with warnings.catch_warnings():
       warnings.simplefilter("ignore", FutureWarning)
       # Explicitly set the number of clusters to the desired value
       kmeans = KMeans(n_clusters=num_colors, random_state=42)
       kmeans.fit(pixels)


   dominant_colors = kmeans.cluster_centers_
   dominant_colors = dominant_colors.astype(int)


   print("Color Palette:")
   for color in dominant_colors:
       hex_value = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
       rgb_value = f"RGB: ({color[0]}, {color[1]}, {color[2]})"
       color_name = rgb_to_closest_color_name((color[0], color[1], color[2]))
       print(f"Hex: {hex_value}, {rgb_value}, Color Name: {color_name}")


   width = 50
   height = 220
   palette_img = Image.new("RGB", (width, height * len(dominant_colors)))
   draw = ImageDraw.Draw(palette_img)


   for i, color in enumerate(dominant_colors):
       draw.rectangle([0, i * height, width, (i + 1) * height], fill=(color[0], color[1], color[2]))


   palette_img.save(output_path)








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
           filename = file.filename
           file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
           temp_file_str = str(filename)
           temp_file_name = remove_extension(file_str)
           temp_input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
           temp_output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "compressed", temp_file_name)
           file_size=500
           compress_image(temp_input_image_path, temp_output_image_path, file_size)
           input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "compressed", temp_file_name)
           


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
 app.run(debug=False)

















