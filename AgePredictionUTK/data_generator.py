import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from HYPERPARAMETERS import IMAGE_PATH
import pandas as pd

class UTKDataset():
    def run(self):
        age, images, gender = self.load_data()
        return self.create_dataframe(age, images, gender)

    def load_data(self):
        images = []
        age = []
        gender = []
        for path_img in os.listdir(IMAGE_PATH):
            age_gender = path_img.split("_")
            ages = int(age_gender[0])
            genders = int(age_gender[1])
            images.append(os.path.join(IMAGE_PATH, path_img))
            # img = cv2.imread(os.path.join(IMAGE_PATH, path_img))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # images.append(img)
            age.append(ages)
            gender.append(genders)
            
        
        age = np.array(age,dtype=np.int64)
        images = np.array(images)   #Forgot to scale image for my training. Please divide by 255 to scale. 
        gender = np.array(gender,np.uint64)

        return age, images, gender

    def create_dataframe(self, age, images, gender):
        x_train, x_test, y_train_age, y_test_age = train_test_split(images, age, random_state=42)
        _, _, y_train_gender, y_test_gender = train_test_split(images, gender, random_state=42)
        # Dataframe age
        train_df_age = pd.DataFrame({'image': x_train, 'age': y_train_age})
        test_df_age = pd.DataFrame({'image': x_test, 'age': y_test_age})
        # Dataframe gender
        train_df_gender = pd.DataFrame({'image': x_train, 'gender': y_train_gender})
        test_df_gender = pd.DataFrame({'image': x_test, 'gender': y_test_gender})

        return train_df_age, test_df_age, train_df_gender, test_df_gender

    # def getitem(self, index, df):
    #     img = Image.open(df.iloc[index]['file'])
    #     label = df.iloc[index]['age']
    #     return img, label



    