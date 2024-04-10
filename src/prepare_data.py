import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf


def prepare_data(config):
    print("Preparing data")
    df = pd.read_csv(config.data.csv_file_path)
    df["label"] = pd.factorize(df["sentiment"])[0]
    #print(df.head())
    #print(df.index)

    #Number of rows:
    print(f"Number of rows in dataframe: {df.shape[0]}")

    train_df, test_df = train_test_split(df, test_size=config.data.test_set_ratio, stratify=df["sentiment"], random_state=1234)

    train_df.to_csv(config.data.train_csv_save_path)
    test_df.to_csv(config.data.test_csv_save_path)



if __name__=="__main__":
    config = OmegaConf.load("../data/params.yaml")
    prepare_data(config)