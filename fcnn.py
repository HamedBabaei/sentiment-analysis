from keras.layers import Dropout, Dense, BatchNormalization,Activation
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras

class FCNN:
    def __init__(self, input_dim, nb_classes, best_model, epoch, 
                 batch_size, validation_split,verbose=0):
        """
        initialize model and compile it
        """
        self.verbose=verbose
        self.best_model=best_model+"_best_model.h5"
        self.epoch=epoch
        self.batch_size=batch_size
        self.validation_split=validation_split
        self.early_stoping=EarlyStopping(monitor='val_accuracy', 
                                           mode='max', 
                                           verbose=1, 
                                           patience=2)
        
        self.model_checkpoint=ModelCheckpoint(self.best_model, 
                                                monitor='val_accuracy', 
                                                mode='max', 
                                                verbose=0, 
                                                save_best_only=True)
        
        self.model = Sequential()
        model_layers = {
            "input":[Dense(512, input_dim=input_dim),BatchNormalization(), 
                           Activation('relu'), Dropout(0.4)],
            "hiddenlayer-1":[Dense(256), Activation('relu')],
            "hiddenlayer-3":[Dense(256), Activation('relu')],
            "hiddenlayer-4":[Dense(256), Activation('relu')],
            "out":[Dense(nb_classes), BatchNormalization(), Activation('sigmoid')] 
        }

        for _, layers in model_layers.items():
            for layer in layers:
                self.model.add(layer)
         
        self.model.compile(loss='sparse_categorical_crossentropy', 
                            optimizer = keras.optimizers.adam(learning_rate=0.002),
                            metrics=['accuracy'])
        
    def fit(self, X_train, y_train):
        """
        Train a model using train/val set 
        Train done with callbacks to save best model
        """
        self.history = self.model.fit(X_train, y_train,
                                      validation_split=self.validation_split,
                                      epochs=self.epoch,
                                      batch_size=self.batch_size,
                                      verbose=self.verbose,
                                      callbacks=[self.early_stoping,
                                                 self.model_checkpoint])
        self.loading_model(self.best_model)

    def loading_model(self, path):
        """Loading pre-trained model"""
        self.model = load_model(path)

    def predict(self, X):
        """Make a prediction using trained model"""
        return self.model.predict_classes(X, verbose=0)
