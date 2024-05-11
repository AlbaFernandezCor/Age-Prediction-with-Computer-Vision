import cv2
from PIL import Image
import numpy as np
from mtcnn import MTCNN
import pickle
import joblib

# load face detector
detector = MTCNN()

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
        center_img = cv2.cvtColor(center_img, cv2.COLOR_BGR2GRAY)
        age_preds = np.mean(age_model.predict(center_img))
        
        # output to the cv2
        return_res.append([top, right, bottom, left, age_preds])
        
    return return_res


if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)
    
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
    
        # Convert the image from BGR color (which OpenCV uses) to RGB color 
        rgb_frame = frame[:, :, ::-1]
    
        # Find all the faces in the current frame of video
        face_locations = detect_face(rgb_frame)
    
        # Display the results
        for top, right, bottom, left, age_preds in face_locations:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
            cv2.putText(frame, 'Age: {:.3f}'.format(age_preds), (left, top-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
                    
        # Display the resulting image
        cv2.imshow('Video', frame)
    
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()