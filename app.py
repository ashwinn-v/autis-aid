from flask import Flask, render_template, url_for
import cv2

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('homepage.html')


from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.optimizers import RMSprop
app=Flask(__name__)
camera = cv2.VideoCapture(1)

#model = load_model('bin_aut.h5')
model = load_model('Mymodel.h5')
#model.compile(optimizer=RMSprop(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def gen_frames():  
    i = 0
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            detector=cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
            #eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
            faces=detector.detectMultiScale(frame,1.1,7)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             #Draw the rectangle around each face
            for (x, y, w, h) in faces:
            
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                if(i%100==0):
                    img = roi_color
                    #img = cv2.imread('')
                    img = cv2.resize(img,(150,150))
                    img = np.reshape(img,[1,150,150,3])
                    #classes = model.predict(img)
                    classes = (model.predict(img) > 0.5).astype("int32")
                    print(classes)

                i = i+1

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                                


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
