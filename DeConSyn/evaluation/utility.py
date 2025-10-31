import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_numerical_columns(df: pd.DataFrame, target: str, categorical_columns: list[str]) -> list[str]:
    exclude = set(categorical_columns + [target])
    return [col for col in df.select_dtypes(include='number').columns if col not in exclude]

class LogisticRegressionEvaluator:
    def __init__(
            self,
            train: pd.DataFrame,
            test: pd.DataFrame,
            target: str,
            categorical_columns: list[str],
            seed: int = 42
    ):
        self.seed = seed

        self.categorical_columns = categorical_columns
        self.numerical_columns = get_numerical_columns(train, target, categorical_columns)

        self.train = train
        self.X_train = train.drop(columns=[target])
        self.y_train = train[target]

        self.test = test
        self.X_test = test.drop(columns=[target])
        self.y_test = test[target]

        self.y_train = self.y_train.astype(str).str.strip().str.replace(r'\.$', '', regex=True)
        self.y_test = self.y_test.astype(str).str.strip().str.replace(r'\.$', '', regex=True)

        self.target = target
        # Encode target if not numeric
        if not pd.api.types.is_numeric_dtype(self.y_train):
            self.label_encoder = LabelEncoder()
            self.y_train = self.label_encoder.fit_transform(self.y_train)
            self.y_test = self.label_encoder.transform(self.y_test)
        else:
            self.label_encoder = None
        self.preprocess()
        self.model = LogisticRegression(max_iter=500, random_state=self.seed)

    def preprocess(self):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_columns),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_columns)
            ]
        )
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor)
        ])

        self.pipeline.fit(self.X_train, self.y_train)
        self.X_train = self.pipeline.transform(self.X_train)
        self.X_test = self.pipeline.transform(self.X_test)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def fit_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """
        Evaluate the logistic regression model on the test set.
        Returns:
            accuracy (float): The accuracy of the model on the test set.
            f1 (float): The weighted F1 score of the model on the test set.
        """
        # If not trained yet, train the model
        if not hasattr(self.model, 'coef_'):
            self.fit_model()
        y_pred = self.model.predict(self.X_test)
        accuracy = self.model.score(self.X_test, self.y_test)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        return accuracy, f1