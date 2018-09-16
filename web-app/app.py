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

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        path = 'static/img/' + filename

        clf = extract_model('adaboost_0.99_0.79.sav')

        # extract labels
        label_map = extract_label_map('labels.txt')

        #test on single image
        tests = []
        img_path = path
        rgb_mean = color_mean_extractor(img_path)
        tests.append(rgb_mean)
        data = [img_path, label_map[clf.predict(tests)[0]].lower()]
        print(label_map[clf.predict(tests)[0]])



        print(path)
        return render_template('picture.html', data = data)
    return render_template('picture.html')

if __name__ == "__main__":
    app.run(debug=True)