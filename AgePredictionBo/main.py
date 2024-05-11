from deep_learning import DeepLearning
from old_school import OldSchoolMethod
from data_generator import CACDDataset
from math import ceil
from prepare_img import ImagePreparing
import pickle

def run():
    print('Iniciant execució...')
    train_complet, test_complet, valid_complet = CACDDataset().run()
    print(len(train_complet), len(test_complet), len(valid_complet))
    train_df, test_df, valid_df = ImagePreparing().run(train_complet, test_complet, valid_complet, 1000)
    print(len(train_df), len(test_df), len(valid_df))
    # train_df = pickle.load(open('C:/Users/Usuario/Documents/Age_Prediction_VC_1/Age_Prediction_VC/train_df.pkl', 'rb'))
    # test_df = pickle.load(open('C:/Users/Usuario/Documents/Age_Prediction_VC_1/Age_Prediction_VC/test_df.pkl', 'rb'))
    # valid_df = pickle.load(open('C:/Users/Usuario/Documents/Age_Prediction_VC_1/Age_Prediction_VC/valid_df.pkl', 'rb'))
    
    # print('Iniciant Old School Method...')
    # OldSchoolMethod().run(train_df, test_df, model_loaded=False)
    # print('Iniciant Deep Learning Method...')
    # DeepLearning().run(train_df, test_df, valid_df)

if __name__ == '__main__':
    run()