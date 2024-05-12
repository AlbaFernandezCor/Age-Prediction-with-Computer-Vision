import cv2
from PIL import Image
import numpy as np
from mtcnn import MTCNN
import joblib
import time

# load face detector
# detector = MTCNN()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# load the model
age_model = joblib.load("AgePredictionBo/models/checkpoints/modelRF_trained.joblib")

def detect_face(img):
    
    mt_res = detector.detect_faces(img)
    return_res = []
    
    for face in mt_res:
        x, y, width, height = face['box']
        center = [x+(width/2), y+(height/2)]
        max_border = max(width, height)
        
        # center alignment
        left = max(int(center[0]-(max_border/2)), 0)
        right = max(int(center[0]+(max_border/2)), 0)
        top = max(int(center[1]-(max_border/2)), 0)
        bottom = max(int(center[1]+(max_border/2)), 0)
        
        # crop the face
        center_img_k = img[top:top+max_border, 
                           left:left+max_border, :]
        center_img = np.array(Image.fromarray(center_img_k).resize([age_model.n_features_in_, age_model.n_features_in_]))
        
        # create predictions
        # center_img = cv2.cvtColor(center_img, cv2.COLOR_BGR2GRAY)
        # age_preds = np.mean(age_model.predict(center_img))
        age_preds = 45
        
        # output to the cv2
        return_res.append([top, right, bottom, left, age_preds])
        
    return return_res


if __name__ == '__main__':
    print('Iniciando...')
    video_capture = cv2.VideoCapture(0)
    tiempo_grabacion = 30
    start_time = time.time()
    while (time.time() - start_time) < tiempo_grabacion:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90))

        # # Convert the image from BGR color (which OpenCV uses) to RGB color 
        # rgb_frame = frame[:, :, ::-1]
    
        # # Find all the faces in the current frame of video
        # face_locations = detect_face(rgb_frame)
    
        # Display the results
        for (x, y, w, h) in faces:
            # Draw a box around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            center_img = np.array(Image.fromarray(frame[y:y+h, x:x+w, :]).resize([age_model.n_features_in_, age_model.n_features_in_]))
            face_img = cv2.cvtColor(center_img, cv2.COLOR_BGR2GRAY)
            age_preds = np.mean(age_model.predict(face_img))
            cv2.putText(frame, 'Age: {:.3f}'.format(age_preds), (x, x+w), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
                    
        # Display the resulting image
        cv2.imshow('Video', frame)
    
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()