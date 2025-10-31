import pandas as pd
from sklearn.model_selection import train_test_split

from DeConSyn.data.data_loader import CARDIO_CATEGORICAL_COLUMNS, CARDIO_TARGET
from DeFeSyn.data.data_loader import DatasetLoader

cardio_path = "../../data/cardio/cardio_train.csv"
cardio = pd.read_csv(cardio_path, sep=";", header=0)
cardio.drop(columns=["id"], inplace=True)
X = cardio.drop(columns=[CARDIO_TARGET])
y = cardio[CARDIO_TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)
train_df.to_csv("../../data/cardio/csv/train.csv", index=False, sep=",")
test_df.to_csv("../../data/cardio/csv/test.csv", index=False, sep=",")