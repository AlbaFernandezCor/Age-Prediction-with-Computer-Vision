from deep_learning import DeepLearning
from old_school import OldSchoolMethod
from data_generator import CACDDataset
from math import ceil


def run():
    print('Iniciant execució...')
    train_df, test_df, valid_df = CACDDataset().run()
    train_df = train_df.head(ceil(len(train_df)*0.003))
    test_df = test_df.head(ceil(len(test_df)*0.003))
    print('Iniciant Old School Method...')
    OldSchoolMethod().run(train_df, test_df, model_loaded=True)
    # DeepLearning().run(train_df, test_df, valid_df)

if __name__ == '__main__':
    run()