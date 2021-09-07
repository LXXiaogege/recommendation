from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer


def main():
    data, label = load_iris(return_X_y=True)

    transformer = Normalizer().fit(data[label != 0])
    X = transformer.transform(data[label != 0])
    y = label[label != 0]
    y[y == 1] = 0
    y[y == 2] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

    clf = LogisticRegression(solver='sag', max_iter=4000)
    clf.fit(X_train, y_train)

    print("score", clf.score(X_test, y_test))


if __name__ == '__main__':
    main()
