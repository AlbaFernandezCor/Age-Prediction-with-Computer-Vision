from deep_learning import DeepLearning
from old_school import OldSchoolMethod
from data_generator import CACDDataset
from math import ceil
from prepare_img import copiar_imagenes


def run():
    print('Iniciant execució...')
    train_df, test_df, valid_df = CACDDataset().run()
    print(len(train_df), len(test_df))
    directorio_destino = "C:/Users/Marina/Documents/GitHub/Uni/Age_Prediction_VC/ImgReduit"
    train_df = train_df.groupby('age', group_keys=False).apply(lambda x: x.sample(frac=0.01))
    copiar_imagenes(train_df.file.unique(), directorio_destino)
    test_df = test_df.groupby('age', group_keys=False).apply(lambda x: x.sample(frac=0.01))
    copiar_imagenes(test.file.unique(), directorio_destino)
    print(len(train_df), len(test_df))
    # print('Iniciant Old School Method...')
    # OldSchoolMethod().run(train_df, test_df, model_loaded=True)
    # DeepLearning().run(train_df, test_df, valid_df)

if __name__ == '__main__':
    run()