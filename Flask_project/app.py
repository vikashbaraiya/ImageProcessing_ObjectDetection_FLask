import io
import os
from io import BytesIO
from flask import Flask, json, render_template, request, jsonify, send_file, send_from_directory, flash, url_for
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_mysqldb import MySQL
import mysql.connector
import base64
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import datetime
from flask_cors import CORS


app = Flask(__name__)

CORS(app)

#connect flask to mysql using mysql.connector
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password='root',
    database="flask",
    port=3306,
    auth_plugin='mysql_native_password'
)

sql = MySQL(app)
#Upload image from pc to upload folder
UPLOAD_FOLDER = 'static/upload'
#processed image is download in this folder download
DOWNLOAD_FOLDER = 'static/download'
#image extention may be in this format
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

#generate a saved file name 
basename = "processImage"
extention = ".jpg"
suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
filename1 = "_".join([basename, suffix]) # e.g. 'mylogfile_120508_171442'
today_datetime = datetime.datetime.today()

#To check image extention in lower case
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER



#this function upload image from pc folder and save in specific server static/upload folder
@app.route('/upload', methods=['POST'])
def upload_media():
    if 'file' not in request.files:
        return jsonify({'error': 'media not provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        cursor = mydb.cursor()
        query = "INSERT INTO upload(filename,file,date) VALUES(%s,%s,%s)"
        print("+++++++++-------",query)
        print("FIle________",file,)
        print("FIlename____",filename )
        
        binaryData = file.read()
        print("binary data",binaryData)
        cursor.execute(query,(file.filename, file.read(),today_datetime))
        mydb.commit()
        cursor.close()
        return jsonify({'msg': 'media uploaded successfully'})

# create download function for download files
@app.route('/download/<filename>')
def download(filename, params={},id=0):
    cursor = mydb.cursor()
    # query1 = "SELECT * FROM images WHERE filename=%s"
    query = "SELECT * FROM images WHERE filename = '"+ str(filename) + "'";
    print("query", query, "IDDDDDDDDDD",filename)
    column_name = (filename,)
    print("column name",column_name)
    cursor.execute(query)
    data = cursor.fetchall()
    for filename in data:
        imagename = filename
        print("imagename",str(imagename[1]))
    print("Data___",data)
    
    return send_file(os.path.join(DOWNLOAD_FOLDER,imagename[1]),
                     download_name=imagename[1], as_attachment=True)

#this function returns list of images names
@app.route('/images', methods=['GET'])
def home():
   image_names = os.listdir('static/upload')
   file = request.files.getlist('files[0]')
   for img in image_names:
       image_array = cv2.imread(os.path.join(UPLOAD_FOLDER,img))
       print(image_array)
   return jsonify({'image_name':image_names,'files':file,"img":img})
 
#Retrieve image from database using specific id
@app.route('/images/<id>', methods=['POST', 'GET'])
def single_image(id):
    cursor = mydb.cursor()
    query = 'SELECT filename, file FROM images WHERE id=%s'
    print("query", query, "IDDDDDDDDDD",id)
    cursor.execute(query,list(id))
    data = cursor.fetchall()
    print("data____",data)
    columnName = ("filename","file")
    for x in data:
            print({columnName[i] :  x[i] for i, _ in enumerate(x)})
                 
              
    # The returned data will be a list of list
    image = data[0][0]
    img = Image.open(os.path.join(UPLOAD_FOLDER,image), mode='r')
    byte_arry = io.BytesIO()
    img.save(byte_arry,format="png")
    encoded_img = base64.encodebytes(byte_arry.getvalue()).decode('ascii')
    print('Image____',img)

    image_path = "/static/upload/" + image
    print("image_path_______",image_path)

    #Decode the string
    #binary_data = base64.b64decode(img)
    #Convert the bytes into a PIL image
    #image = Image.open(io.BytesIO(img))
    #image.show()
    
    print("object______",img)

    return jsonify({'image':str(img),"image_name":image,"encoded_image":encoded_img, "path":image_path})

#edit image and save it to download folder 
@app.route("/edit", methods=['POST', 'GET'])
def edit():
    if request.method == "POST":
        operation = request.form.get("operation")
        if 'file' not in request.files:
             return jsonify({'error': 'media not provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'no file selected'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #processImage function call 
            processImage(filename, operation="edgeDetection")
            flash(f"Your image has been processed and is available <a href='/download/{filename}' target='_blank'>here</a>")
            return jsonify({'msg': 'media uploaded successfully'})
    
  
#edit image using opencv and take two perameter filename and operation
# and use in edit function 
def processImage(filename, operation):
    print(f"the operation is {operation} and filename is {filename}")
    img = cv2.imread(f'static/upload/{filename}')
    print("image-----:",img)
    #user can choice operation to edit uploaded image
    match operation:
        case "cgray":
            imgProcessed = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            newFilename = f"static/download/{filename}"
            print("imageProcess___________", imgProcessed)
            cv2.imwrite(newFilename, imgProcessed)
            return newFilename
        case "addText":
            
            i = 0
            while True:
               # Display the image
               cv2.imshow('a',img)
               # wait for keypress
               k = cv2.waitKey(0)
               # specify the font and draw the key using puttext
               font = cv2.FONT_HERSHEY_SIMPLEX
               addText = cv2.putText(img,chr(k),(140+i,250), font, .5,(255,255,255),2,cv2.LINE_AA)
               i+=10
               if k == ord('q'):
                   break
            newFilename = f"static/download/{filename}"
            cv2.imwrite(newFilename,addText)
            #addText.save(os.path.join(app.config['DOWNLOAD_FOLDER'], img))
            cv2.destroyAllWindows()
            return newFilename
        case "resize":
            newFilename = f"static/upload/{filename}"
            filename = "_".join([basename, suffix, extention])
            print("filename1",filename)
            img = cv2.imread(newFilename, cv2.IMREAD_UNCHANGED)
            print('Original Dimensions : ',img.shape)
 
            width = 350 #int(request.form['width'])
            height = 450 #int(request.form['height'])
            dim = (width, height)
 
            # resize image
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
            print('Resized Dimensions : ',resized.shape)
            newFilename = f"static/download/{filename}"
            cv2.imwrite(newFilename,resized)
            cv2.destroyAllWindows()
            return newFilename
        
        case "edgeDetection":
            newFilename = f"static/upload/{filename}"
            filename = "_".join([basename, suffix, extention])
            img = cv2.imread(newFilename)
            edges = cv2.Canny(img, 100,200)
            newFilename = f"static/download/{filename}"
            cv2.imwrite(newFilename,edges)
            cv2.destroyAllWindows()
            return newFilename

@app.route("/filter", methods=['POST', 'GET'])
def imageBlur():
    if request.method == "POST":
        operation = request.form.get("operation")
        if 'file' not in request.files:
             return jsonify({'error': 'media not provided'}), 400
        file = request.files['file']
        print("File_______",file)
        if file.filename == '':
            return jsonify({'error': 'no file selected'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #processImage function call 
            blurImageFilter(filename, operation="sobel")
            flash(f"Your image has been processed and is available <a href='/download/{filename}' target='_blank'>here</a>")
            return jsonify({'msg': 'media uploaded successfully'})
    
def blurImageFilter(filename, operation):
    print(f"the operation is {operation} and filename is {filename}")
    img = cv2.imread(f'static/upload/{filename}')
    print("image-----:",img)
    #user can choice operation to edit uploaded image
    match operation:
        case "medianBlur":
            kernal_size = 5 #int(request.form['kernal_size'])
            blur = cv2.medianBlur(img, kernal_size) 
            newFilename = f"static/download/{filename}"
            print("imageProcess___________", blur)
            cv2.imwrite(newFilename, blur)
            cv2.destroyAllWindows()
            return newFilename
        case "guassianBlur":
            k1 = 5 #int(request.form['k1'])
            k2 = 5 #int(request.form['k2'])
            #k1 and k2 value should be same and odd
            blur = cv2.GaussianBlur(img, (k1,k2), 2, cv2.BORDER_DEFAULT)
            newFilename = f"static/download/{filename}"
            print("blur",blur)
            cv2.imwrite(newFilename,blur)
            cv2.destroyAllWindows()
            return newFilename
        case "sobel":
            # Convert to graycsale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Blur the image for better edge detection
            img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
            axis = "xy_axis" #request.form['axis']
            # Sobel Edge Detection
            if axis=='laplascian':
               laplacian = cv2.Laplacian(img,cv2.CV_64F, ksize=5)
               newFilename = f"static/download/{filename}"
               cv2.imwrite(newFilename,laplacian)
            elif axis=='x_axis':
               sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
               newFilename = f"static/download/{filename}"
               cv2.imwrite(newFilename,sobelx)
            elif axis=='y_axis':
               sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
               newFilename = f"static/download/{filename}"
               cv2.imwrite(newFilename,sobely)
            elif axis=='xy_axis':
               sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
               newFilename = f"static/download/{filename}"
               cv2.imwrite(newFilename,sobelxy)
            # Display Sobel Edge Detection Images

            cv2.destroyAllWindows()
            return newFilename
                        

#crop image and download in download folder
@app.route("/crop", methods=["POST"])
def crop():
    # retrieve parameters from html form
    x1 = 100 #int(request.form['x1'])
    y1 = 100 #int(request.form['y1'])
    x2 = 500 #int(request.form['x2'])
    y2 = 500 #int(request.form['y2'])
    #filename = request.form['image']
    file = request.files['file']
    filename = secure_filename(file.filename)
    img = cv2.imread(f'static/upload/{filename}')
    print("immage",img)

    # open image
    target =  f'static/upload'
    destination = "/".join([target, filename])

    img = Image.open(destination)

    # check for valid crop parameters
    width = img.size[0]
    height = img.size[1]
    print("widht:",width, "height:",height)

    crop_possible = True
    if not 0 <= x1 < width:
        crop_possible = False
    if not 0 < x2 <= width:
        crop_possible = False
    if not 0 <= y1 < height:
        crop_possible = False
    if not 0 < y2 <= height:
        crop_possible = False
    if not x1 < x2:
        crop_possible = False
    if not y1 < y2:
        crop_possible = False

    # crop image and show
    if crop_possible:
        img = img.crop((x1, y1, x2, y2))
        
        # save and return image
        target1 =  f'static/download'
        filename = "_".join([basename, suffix, extention])
        destination = "/".join([target1, filename])
        print("destination____",destination)
        if os.path.isfile(destination):
            os.remove(destination)
        img.save(destination)
        return send_image(filename)
    else:
        return jsonify({"message":"Crop dimensions not valid"}),400
    return jsonify({'msg':'Crop Done'}),204
    
# flip filename 'vertical' or 'horizontal'
@app.route("/flip", methods=["POST"])
def flip():

    # retrieve parameters from html form
    mode1 = "horizontal"
    if 'horizontal' in mode1:    #request.form['mode']
        mode = 'horizontal'
        print("mode:",mode)
    elif 'vertical' in request.form['mode']:
        mode = 'vertical'
    else:
        return jsonify({ "message":"Mode not supported (vertical - horizontal)"}), 400
    #filename = request.form['image']
    file = request.files['file']
    filename = secure_filename(file.filename)
    print("filename", filename)

    # open and process image
    target = f'static/upload'
    destination = "/".join([target, filename])

    img = Image.open(destination)
    print("img",img)

    if mode == 'horizontal':
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        print('img__hori',img)
    else:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # save and return image
    target1 = f'static/download'
    filename = "_".join([basename, suffix, extention])
    destination = "/".join([target1, filename])
    if os.path.isfile(destination):
        os.remove(destination)
    img.save(destination)

    return send_image(filename)

# rotate filename the specified degrees
@app.route("/rotate", methods=["POST"])
def rotate():
    # retrieve parameters from html form
    angle = 90 #request.form['angle']
    file = request.files['file']
    filename = secure_filename(file.filename)
    #filename = request.form['image']

    # open and process image
    target = f'static/upload'
    destination = "/".join([target, filename])

    img = Image.open(destination)
    img = img.rotate(-1*int(angle))

    # save and return image
    target1 = f'static/download'
    filename = "_".join([basename, suffix, extention])
    destination = "/".join([target1, filename])
    if os.path.isfile(destination):
        os.remove(destination)
    img.save(destination)

    return send_image(filename)



# retrieve file from 'static/upload' directory
@app.route('/static/download/<filename>' , methods=['POST'])
def send_image(filename):
    print("sendimage directory______")
    
    return send_from_directory(DOWNLOAD_FOLDER, filename)
#Retrieve all images from upload folder with filename and stored image path
@app.get("/img")
def images():
    out = []
    for filename in os.listdir("static/upload"):
        out.append({
            "name": filename.split(".")[0],
            "path": "/static/upload/" + filename
        })
    return jsonify({"image_data":out})

#object detection function to detect uploaded image object name
@app.route("/objectDetection", methods=["POST","GET"])
def objectDetection():
    file = request.files['file']
    filename = secure_filename(file.filename)

   #  file = request.files['file']
   #  filename = secure_filename(file.filename)
    image = cv2.imread(f'static/upload/{filename}')
    print("image : ",image)
    #image = cv2.imread(destination)
    image = cv2.resize(image, (640,420))
    h = image.shape[0]
    w = image.shape[1]

    # path to the weights and model files
    weights = "frozen_inference_graph.pb"
    model = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    # load the MobileNet SSD model trained  on the COCO dataset
    net = cv2.dnn.readNetFromTensorflow(weights, model)

    # load the class labels the model was trained on
    class_names = []
    with open("objectname.txt", "r") as f:
       class_names = f.read().strip('\n').split("\n")
       print(class_names)

    # create a blob from the image
    blob = cv2.dnn.blobFromImage(
       image, 1.0/127.5, (320, 320), [127.5, 127.5, 127.5])
    # pass the blog through our network and get the output predictions
    net.setInput(blob)
    output = net.forward() # shape: (1, 1, 100, 7)

    # loop over the number of detected objects
    for detection in output[0, 0, :, :]: # output[0, 0, :, :] has a shape of: (100, 7)
       # the confidence of the model regarding the detected object
       probability = detection[2]

       # if the confidence of the model is lower than 50%,
       # we do nothing (continue looping)
       if probability < 0.5:
          continue

       # perform element-wise multiplication to get
       # the (x, y) coordinates of the bounding box
       box = [int(a * b) for a, b in zip(detection[3:7], [w, h, w, h])]
       box = tuple(box)
       # draw the bounding box of the object
       cv2.rectangle(image, box[:2], box[2:], (0, 255, 0), thickness=2)

       # extract the ID of the detected object to get its name
       class_id = int(detection[1])
       # draw the name of the predicted object along with the probability
       label = f"{class_names[class_id - 1].upper()} {probability * 100:.2f}%"
       cv2.putText(image, label, (box[0], box[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    #download object detected image in static/download folder
    target1 = f'static/download'
    filename = "_".join([basename, suffix, extention])
    destination = "/".join([target1, filename])
    if os.path.isfile(destination):
        os.remove(destination)
    #image.save(destination)
    
    cv2.imwrite(destination,image)

    return send_image(filename)


#feature detect on two image function using sift model 
@app.route('/featuredetection', methods=['POST', 'GET'])
def featureDetection():
      file = request.files['file']
      files = request.files.getlist("file")
      print("files",files)
      listname = []
      for file in files:
         filename = secure_filename(file.filename)
         listname.append(filename)
      filename1 = listname[0]
      filename2 = listname[1]
         
      print("filename1", filename, "filename2:",filename[1])
  
      # read images
      img1 = cv2.imread(f'static/upload/{filename1}')
      print("img1",img1) 
      img2 = cv2.imread(f'static/upload/{filename2}') 
      print("img2",img2) 

      img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
      img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
   
      #sift
      sift = cv2.SIFT_create()

      keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
      keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

      #feature matching
      bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

      matches = bf.match(descriptors_1,descriptors_2)
      matches = sorted(matches, key = lambda x:x.distance)

      img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
      plt.imshow(img3),plt.show()
      #download object detected image in static/download folder
      target1 = f'static/download'
      filename = "_".join([basename, suffix, extention])
      destination = "/".join([target1, filename])
      if os.path.isfile(destination):
          os.remove(destination)
      #image.save(destination)
    
      cv2.imwrite(destination,img3)

      return send_image(filename),jsonify({'msg':"image saved successfully"})
      
def search(params={}):
    cursor = mydb.cursor()
    filename = params.get("filename", None)
    print('filename___', filename)
    query = "SELECT * FROM images WHERE filename = '"+ str(filename) + "'";
    print("query", query, "IDD",id)
    cursor.execute(query)
    data = cursor.fetchall()
    print("data____",data)
    columnName = ("id","filename","file", "date")

    res = {
            "data": [],
            "MaxId":1,
        }
    count = 0
    for x in data:
        # print({columnName[i] :  x[i] for i, _ in enumerate(x)})
        print(x)
        res["data"].append({columnName[i]:  x[i] for i, _ in enumerate(x)})
    return res
              

@app.route('/search', methods=['POST', 'GET'])
def searching(params={}):
    file = request.files['file']
    params["filename"] = secure_filename(file.filename)
    print('param', params['filename'])
    json_request = request.get_data
    print("json_request",json_request)
    params["filename1"] = request.form.get("filename",None)
    print("paramFilenael",params["filename1"])
    params["file"] = request.form.get("file",None)
    print('searching_params', params)
    c = search(params)
    print("searching",c)
    res = { "mesg":"","message":""}
    if (c != None):
        file = c["data"][0]['file'].decode("utf-8")
        c["data"][0]['file'] = file
        res["data"] = c["data"]
        if res["data"] == []:
            res["mesg"] = "No record found"
        res["error"] = False
    else:
        res["error"] = True
        res["message"] = "No record found"
    print("RES_____+__+_+_+_+_= ",res)
    return jsonify({"result":res})
    
    

if __name__ == '__main__':
    app.run(debug=True, port=5000)


   