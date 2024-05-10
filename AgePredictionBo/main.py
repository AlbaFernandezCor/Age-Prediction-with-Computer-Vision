from deep_learning import DeepLearning
from old_school import OldSchoolMethod
from data_generator import CACDDataset
from math import ceil
from prepare_img import ImagePreparing


def run():
    print('Iniciant execució...')
    train_complet, test_complet, valid_complet = CACDDataset().run()
    print(len(train_complet), len(test_complet), len(valid_complet))
    train_df, test_df, valid_df = ImagePreparing().run(train_complet, test_complet, valid_complet, 1000)
    print(len(train_df), len(test_df), len(valid_df))
    # print('Iniciant Old School Method...')
    # OldSchoolMethod().run(train_df, test_df, model_loaded=True)
    # DeepLearning().run(train_df, test_df, valid_df)

if __name__ == '__main__':
    run()