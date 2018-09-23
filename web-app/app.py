from flask import Flask,render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
from sklearn.ensemble import AdaBoostClassifier
import cv2
import sys
import pickle

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
    import numpy as np
    from skimage.io import imread

    # read image from path
    # im = imread('dataset/COLOR/ORANGE/281620453_detail.jpg') <-- this can't be right
    im = imread(image_path)

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
            rgb_list.append(avg)

    return rgb_list, im


def make_predictions(rgb_list, model_path, label_map_path,im):
    '''
    extract pickled model and load
    run classification for any pictures in directory
    potentially show rgb graph
    '''
    
    # extract model
    clf = extract_model(model_path)

    # extract labels
    label_map = extract_label_map(label_map_path)

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


""" @app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        path = 'static/img/' + filename

        ### OLD CLASSIFIER
        #clf = extract_model('adaboost_0.99_0.79.sav')
        # extract labels
        #label_map = extract_label_map('labels.txt')
        #test on single image
        # tests = []
        # img_path = path
        # rgb_mean = color_mean_extractor(img_path)
        # tests.append(rgb_mean)
        # data = [img_path, label_map[clf.predict(tests)[0]].lower()]
        # print(label_map[clf.predict(tests)[0]])

        model_path = 'adaboost_0.99_0.79.sav'
        label_path = 'labels.txt'
        img_path = path
        window_size = 3

        #extract rgb list
        rgb_list,image = extract_windows(img_path,window_size,window_size)

        #run prediction
        final_out, color_store = make_predictions(rgb_list,model_path,label_path,image)
        data = [img_path,final_out]
        
        #sort prediction colors
        labels = extract_label_map(label_path)
        # print(labels)
        # print(color_store)

        temp_store = {}
        array=[]
        array.append(["Color","Frequency"])
        # array.append(['yo',12])
        # array.append(['ro',13])
        # array.append(['to',16])

        temp_store = {}
        for i in color_store:
            temp_store[i[0]] = i[1]
        print(temp_store)
        for i in labels:
            if labels[i] in temp_store:
                array.append([labels[i],temp_store[labels[i]]])
            else:
                array.append([labels[i],0])
        
        data.append(array)
        print(data[2])
        print(img_path)
        print(final_out)
        return render_template('picture.html', data = data)
    return render_template('picture.html') """

@app.route('/upload', methods=['POST'])
def upload():
    """ allows the user to upload multiple images and be tested by all of Ritwiks models """
    uploaded_files = request.files.getlist("photo")
    print("uploaded files: " + str(uploaded_files))
    full_data = []
    for file in uploaded_files:
        filename = photos.save(file)
        path = 'static/img/' + filename
        model_path = 'adaboost_0.99_0.79.sav'
        label_path = 'labels.txt'
        img_path = path
        window_size = 3

        #extract rgb list
        rgb_list, image = extract_windows(img_path, window_size, window_size)

        #run prediction
        final_out, color_store = make_predictions(rgb_list,model_path,label_path,image)
        data = [img_path,final_out]
        
        #sort prediction colors
        labels = extract_label_map(label_path)

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
        full_data.append(data)
    print("Full Data: " + str(full_data))
    return render_template('picture.html', data = full_data)


if __name__ == "__main__":
    app.run(debug=True)