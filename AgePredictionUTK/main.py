from data_generator import UTKDataset
from prepare_img import ImagePreparing
import tensorflow as tf
from old_school import OldSchoolMethod
from deep_learning import DeepLearning

def run():
    print('Creem els dataframes')
    train_df_age_complet, test_df_age_complet, train_df_gender, test_df_gender = UTKDataset().run()
    print("Creem un set reduit d'imatges")
    train_df_age, test_df_age = ImagePreparing().run(train_df_age_complet, test_df_age_complet, df_type = 'age', num_img = 1000)
    # train_df_gender, test_df_gender = ImagePreparing().run(train_df_gender, test_df_gender, df_type = 'gender', num_img = 1000)

    with tf.device('GPU'):
        # OldSchoolMethod().run(train_df_age, test_df_age, model_loaded = False, df_type='age')
        DeepLearning().run(train_df_age, test_df_age, df_type='age')
    print("hola")


if __name__ == '__main__':
    run()