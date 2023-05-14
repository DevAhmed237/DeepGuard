import os
from werkzeug.utils import secure_filename
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import cv2
from mtcnn import MTCNN

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = '123'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}
model = load_model(r'MesoInception_DF.h5')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('not file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('no image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img = load_img(file_path, target_size=(256, 256))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        p = model.predict(x)[0]
        c = f'{p} Fake' if p < 0.5 else f'{p} Real'
        res = {'class_label:', str(c), 'class_probability:', str(p)}

        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename ,res=res)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
    return redirect(request.url)


@app.route('/predict_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No video selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error loading video file")
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
        detector = MTCNN()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image)
        bounding_box = results[0]['box']
        margin_x = bounding_box[2] * 0.3  # 30% as the margin
        margin_y = bounding_box[3] * 0.3  # 30% as the margin
        x1 = int(bounding_box[0] - margin_x)
        if x1 < 0:
            x1 = 0
        x2 = int(bounding_box[0] + bounding_box[2] + margin_x)
        if x2 > image.shape[1]:
            x2 = image.shape[1]
        y1 = int(bounding_box[1] - margin_y)
        if y1 < 0:
            y1 = 0
        y2 = int(bounding_box[1] + bounding_box[3] + margin_y)
        if y2 > image.shape[0]:
            y2 = image.shape[0]
        print(x1, y1, x2, y2)
        crop_image = image[y1:y2, x1:x2]
        img_path = video_path + '.png'
        cv2.imwrite(img_path, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
        cap.release()

        img = load_img(img_path, target_size=(256, 256))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        p = model.predict(x)[0]
        c = f'{p} Fake' if p < 0.5 else f'{p} Real'
        res = {'class_label:', str(c), 'class_probability:', str(p)}

        flash('Video successfully uploaded and processed')
        return render_template('index.html', filename=filename, res=res)
    else:
        flash('Allowed video types are - mp4, avi')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == '__main__':
    app.run()