import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

def get_clean_data():
    data = pd.read_csv(r'model/reduced_data.csv', index_col=0)

    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })

    return data

def create_model(data):
    # split features and target
    x = data.drop(columns=['diagnosis'])
    y = data['diagnosis']

    # split train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # scale the data
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # build and train model
    model = LogisticRegression(
        penalty='l2',
        fit_intercept=False,
        C=0.2
    )
    model.fit(x_train, y_train)

    # test model
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_pred, y_test)
    print('Accuracy: ', acc)

    return model, scaler

def main():
    #import and clean dataset
    data = get_clean_data()

    #create the model
    model, scaler = create_model(data)

    with open(r'model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(r'model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()