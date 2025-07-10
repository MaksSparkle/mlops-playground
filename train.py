from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

def main():
    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier().fit(X, y)
    dump(clf, "model.joblib")

if __name__ == "__main__":
    main()