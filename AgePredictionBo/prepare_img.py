from HYPERPARAMETERS import REDUCED_PATH, IMAGE_PATH
import shutil
import pickle
import os

class ImagePreparing():
    def run(self, train_df, test_df, valid_df, num_img = 1000):
        percent_of_dfs = num_img/len(train_df)
        train_df = self.get_random_img(train_df, percent_of_dfs)
        test_df = self.get_random_img(test_df, percent_of_dfs)
        valid_df = self.get_random_img(valid_df, percent_of_dfs)
        self.save_df(train_df, test_df, valid_df)
        return train_df, test_df, valid_df

    def get_random_img(self, df, percent_of_dfs):
        df =  df.groupby('age', group_keys=False).apply(lambda x: x.sample(frac=percent_of_dfs))
        return self.copiar_imagenes(df)


    def copiar_imagenes(self, df):
        if not os.path.exists(REDUCED_PATH):
            os.makedirs(REDUCED_PATH)
        
        for img_name in df.file:
            cacd_img = os.path.join(IMAGE_PATH, img_name)
            if os.path.isfile(cacd_img):
                shutil.copy(cacd_img, REDUCED_PATH)
                df.loc[df['file']==img_name,'file'] = os.path.join("/content/ImgRed/", img_name)
        
        return df
    
    def save_df(self, train_df, test_df, valid_df):
        shutil.make_archive('ImgReduit', 'zip', REDUCED_PATH)
        pickle.dump(train_df, open("train_df.pkl","wb"))
        pickle.dump(test_df, open("test_df.pkl","wb"))
        pickle.dump(valid_df, open("valid_df.pkl","wb"))


