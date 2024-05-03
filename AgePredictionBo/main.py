from deep_learning import DeepLearning
from data_generator import CACDDataset


def run():
    train_df, test_df, valid_df = CACDDataset().run()
    DeepLearning().run(train_df, test_df, valid_df)

if __name__ == '__main__':
    run()