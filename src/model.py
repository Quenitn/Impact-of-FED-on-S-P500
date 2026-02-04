import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from src.config import RF_ESTIMATORS, RANDOM_STATE, TEST_SIZE

class StrategyModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=RANDOM_STATE)

    def prepare_data(self, df: pd.DataFrame):
        df['target'] = (df['market_return'] > 0).astype(int)
        X = df[['finbert_score']]
        y = df['target']
        return X, y

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)
        self.model.fit(X_train, y_train)
        return X_test, y_test

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]
        print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
        print(f"ROC-AUC  : {roc_auc_score(y_test, probs):.4f}")
        print(classification_report(y_test, preds))