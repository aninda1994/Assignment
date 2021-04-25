#Auhtor:Aninda Kundu

def main():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LeakyReLU, PReLU, ELU
    from keras.layers import Dropout
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score

    #Data preprocessing
    df = pd.read_csv("data.csv")
    df.drop('Unnamed: 32', axis=1, inplace=True)
    df.drop('id', axis=1, inplace=True)
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    labelencoder_X_1 = LabelEncoder()
    y = labelencoder_X_1.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #Check the distributions of all features
    '''
    for feature in X.columns:
        sns.distplot(X[feature])
        plt.ylabel(feature)
        plt.xlabel(feature)
        plt.show()
    '''

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu', input_dim=30))
    # classifier.add(Dropout(0.5))

    # Adding the second hidden layer
    classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu'))
    # classifier.add(Dropout(0.5))
    # Adding the output layer
    classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs=150)

    # list all data in history

    print(model_history.history.keys())
    # summarize history for accuracy
    plt.plot(model_history.history['acc'])
    plt.plot(model_history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)

    # Making the Confusion Matrix

    cm = confusion_matrix(y_test, y_pred)

    # Calculate the Accuracy

    score = accuracy_score(y_pred, y_test)
    print(cm)
    print(score)
    sns.heatmap(cm, annot=True)
    plt.savefig('h.png')


if __name__=="__main__":
    main()