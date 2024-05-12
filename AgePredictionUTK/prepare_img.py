import pickle

class ImagePreparing():
    def run(self, train_df, test_df, df_type = 'age', num_img = 1000): # df_type: age, gender
        percent_of_dfs = num_img/len(train_df)
        train_df = self.get_random_img(train_df, percent_of_dfs, df_type)
        train_df = train_df.reset_index(drop=True)
        test_df = self.get_random_img(test_df, percent_of_dfs, df_type)
        test_df = test_df.reset_index(drop=True)
        self.save_df(train_df, test_df)
        return train_df, test_df

    def get_random_img(self, df, percent_of_dfs, df_type):
        df =  df.groupby(df_type, group_keys=False).apply(lambda x: x.sample(frac=percent_of_dfs))
        return df
 
    def save_df(self, train_df, test_df):
        pickle.dump(train_df, open("train_df.pkl","wb"))
        pickle.dump(test_df, open("test_df.pkl","wb"))
