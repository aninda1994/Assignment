#Author: Aninda Kundu

def main():
    import pandas as pd
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    from sklearn.preprocessing import LabelEncoder
    from tensorflow import keras
    from tensorflow.keras import layers
    from kerastuner.tuners import RandomSearch
    from sklearn.model_selection import train_test_split

    # for gpu initialization
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    #Data Preprocessing
    df = pd.read_csv("data.csv")
    df.drop('Unnamed: 32', axis=1, inplace=True)
    df.drop('id', axis=1, inplace=True)
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    labelencoder_X_1 = LabelEncoder()
    y = labelencoder_X_1.fit_transform(y)

    #using keras tuner select the number of layers and neurons
    def build_model(hp):
        model = keras.Sequential()
        for i in range(hp.Int('num_layers', 2, 20)):
            model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                                min_value=32,
                                                max_value=512,
                                                step=32),
                                   activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='binary_crossentropy',
            metrics=['acc'])
        return model

    tuner = RandomSearch(
        build_model,
        objective='val_acc',
        max_trials=5,
        executions_per_trial=3,
        directory='project',
        project_name='breast cancer3')

    tuner.search_space_summary()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    tuner.search(X_train, y_train,
                 epochs=150,
                 validation_data=(X_test, y_test))

if __name__=="__main__":
    main()





