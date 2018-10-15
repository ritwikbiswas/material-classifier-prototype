from flask import Flask,render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
from sklearn.ensemble import AdaBoostClassifier
# import tensorflow as tf
import cv2
import sys
import pickle
import numpy as np
import numpy as np
from skimage.io import imread
import requests
import json
import ast
import collections
# from skimage.transform import resize
from scipy import misc
# read image from path

app = Flask(__name__)

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about.html")
def about():
    return render_template("about.html")


# @app.route("/PAGE.html")
# def NAME():
#     return render_template("PAGE.html")

################ COLOR MODEL FUNCTION ###################

def extract_model(filename):
    '''
    Extract pickled model
    '''
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


def extract_label_map(filename):
    '''
    Extract label as pickle
    '''
    label_map = {}
    with open(filename,'r') as flabel:
        for line in flabel:
            spl = line.split(",")
            label_map[int(spl[0])] = spl[1][:-1]
    return label_map


def color_mean_extractor(image_location):
    ''' 
    extract mean rgb array and returns in the form [R,G,B]
    '''
    img = cv2.imread(image_location)
    color_mean = cv2.mean(img)
    rgb_color_mean = [color_mean[2], color_mean[1], color_mean[0]]
    return rgb_color_mean


def extract_windows(image_path, window_width, window_height):
    '''
    extracts rgb values of image broken into windows of width x heigh
    '''

    im = imread(image_path)
    print(str("\nDIMENSIONS: " + str(im.shape[0]) + "," + str(im.shape[1])))
    
    #determine resizing of image
    output_width = 300
    if im.shape[1] > output_width:
        scale_factor = int(im.shape[1])/output_width
        im = misc.imresize(im,(int(im.shape[0] / scale_factor), int(im.shape[1] / scale_factor)))
        #misc.imshow(im)
    print(str("\nPOST-DIMENSIONS: " + str(im.shape[0]) + "," + str(im.shape[1])))
    # set block size to 3x3
    wnd_r = window_width
    wnd_c = window_height

    rgb_list = []

    # split image into blocks and compute block average
    for r in range(0, im.shape[0] - wnd_r, wnd_r):
        col = 0
        for c in range(0, im.shape[1] - wnd_c, wnd_c):
            window = im[r:r + wnd_r, c:c + wnd_c]
            avg = np.mean(window, axis=(0, 1))
            if isinstance(avg, np.float64):
                avg = [avg,avg,avg]
            rgb_list.append(avg[:3])

    return rgb_list, im


def make_predictions(rgb_list, clf, label_map,im):
    '''
    extract pickled model and load
    run classification for any pictures in directory
    potentially show rgb graph
    '''
    


    #test on single image
    prediction_list = clf.predict(rgb_list)

    #hash of color prediction
    color_store = {}
    for i in prediction_list:
        name = label_map[i]
        if name in color_store:
            color_store[name] += 1
        else:
            color_store[name] = 1
    
    total_size = len(rgb_list)
    sorted_color_store = sorted(color_store.items(), key=lambda kv: kv[1], reverse=True)

    #Determine majority colors
    percent_total = 0
    final_string = ""
    for i in sorted_color_store:
        final_string = final_string + i[0] + ", "
        temp = float(i[1]/total_size)
        percent_total += temp
        print(temp)
        if percent_total >= 0.8:
            break
    final_string = final_string[:-2]
    print(final_string)
    print(percent_total)

    return final_string, sorted_color_store
    # f = plt.figure()
    # f.suptitle(final_string, fontsize=20)
    # f.add_subplot(1, 2, 1).axis("off")
    # plt.imshow(im)
    # f.add_subplot(1, 2, 2)
    # plt.bar(list(color_store.keys()), color_store.values(), color='grey')
    # plt.xticks(rotation=90)
    # plt.show(block=True)

################ PATTERN MODEL FUNCTIONS ################
def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    return result


@app.route('/upload', methods=['POST'])
def upload():
    
    """ allows the user to upload multiple images and be tested by all of Ritwiks and Davids models """
    ## COLOR MODEL AND LABEL EXTRACTION
    uploaded_files = request.files.getlist("photo")
    print("uploaded files: " + str(uploaded_files))
    full_data = []
    
    #model_path = 'adaboost_0.99_0.79.sav'
    model_path = 'adaboost_new_1.0_0.98.sav'
    label_path = 'labels.txt'
    # extract model
    clf = extract_model(model_path)

    # extract labels
    label_map = extract_label_map(label_path)

    for file in uploaded_files:
        
        filename = photos.save(file)
        path = 'static/img/' + filename
        
        img_path = path
        print(img_path)

        ### PATTERN CLASSIFICATION ###
        pattern_type = "STYLE"
        try:
            url = 'http://pattern-classifier-env.gwms7amgnm.us-east-1.elasticbeanstalk.com/'
            data = {'file': open(img_path, 'rb')}
    
            #send request to classifier and return request classificaiton
            r = requests.post(url, files=data)
            obj = r.json()
            pattern_type = str(obj['img1']['label'])
        except:
            pattern_type = "EMPTY STYLE"


        ### COLOR CLASSIFICATION
        window_size = 7

        #extract rgb list
        rgb_list, image = extract_windows(img_path, window_size, window_size)

        #run prediction
        final_out, color_store = make_predictions(rgb_list,clf,label_map,image)
        data = [img_path,final_out.lower()]
        
        #sort prediction colors
        labels = label_map

        temp_store = {}
        array=[]
        array.append(["Color", "Frequency"])

        temp_store = {}
        for i in color_store:
            temp_store[i[0]] = i[1]
        #print(temp_store)
        for i in labels:
            if labels[i] in temp_store:
                array.append([labels[i],temp_store[labels[i]]])
            else:
                array.append([labels[i],0])
        
        data.append(array)
        
        data.append(pattern_type)
        full_data.append(data)

    # sess.close()
    # print(r)
    print("Full Data: " + str(full_data))
    return render_template('picture.html', data = full_data)

@app.route('/recommendation', methods=["POST"])
def recommendation():
    data = dict(request.form)   #this is the {{item}} passed from picture
    del data['action']          #removes extra, unneccessary tag
    for key in data.keys():     # there should only be one key but need to loop
        item = key              # item contains all the information about the fabric
    item = ast.literal_eval(item)

    #Create request for Rec Engine
    #logic to only include top 3 colors
    top_n = 7
    top_colors = []
    colors = item[2][1:]
    colors.sort(key=lambda k: (k[1]), reverse=True)
    for i in range(0,top_n):
        top_colors.append(colors[i][0])
    
    #count total to give percentage
    total = 0
    colors
    for i in colors:
        print(i)
        i[0] = i[0].lower().capitalize()
        total += int(i[1])

    #store color dict with only top n values
    color_dict = {}
    for i in colors:
        if top_n == 0:
            color_dict[i[0]] = 0
        else:
            color_dict[i[0]] = i[1]/total
            top_n -= 1

    #sort dictionary in alphabetical
    color_dict = collections.OrderedDict(sorted(color_dict.items()))

    #build request
    rec_request = {}
    rec_request['Pattern'] = item[len(item)-1].capitalize()
    rec_request['Color Distribution'] = dict(color_dict)
    data_json = json.dumps(rec_request)

    #print(data_json)

    #Send JSON request
    r = requests.post('https://kr4h95boel.execute-api.us-east-2.amazonaws.com/production_query', data=data_json)
    obj = r.json()
    # print(obj)
    
    recommendations = []
    #empty recommendations
    for i in range(0,4):
        recommendations.append(['No Recommendation',100,'http://dentdelion.net/picture/2018/04/Used-Office-Furniture-Tri-State-Office-Furniture.jpg','https://www.hermanmiller.com/'])
    
    count = 0
    for entry in obj:
        if count == 4:
            break
        temp = [entry[2],entry[3],entry[0],entry[1]]
        recommendations[count] = temp
        count += 1
    recommendations.append([item[0],item[1],item[len(item)-1]])
    print(recommendations)
    return render_template('recommendation.html', data = recommendations)


if __name__ == "__main__":
    app.jinja_env.cache = {}
    app.run(debug=True)