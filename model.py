
class ModelPipeline:
    def __init__(self, estimator=None, transformer=None, 
                       dim_reduction=None, verbose=False):
        """"
          :transformer, could be a class or any thing, 
                        only it need to have these methods
                         :fit, for learning the transformer
                         :transform, for transforming given input!
          :estimator, your classification model, 
                      or a personalized class for classification, 
                      and it could be anything for classification of given data,
                      it must have these methods,
                         :fit, for learning a estimator
                         :predict, for predicting a given inputs
                         :predict_proba, to provide a probability instead of class
          :dim_reduction, your dimension reduction technique,
                      it could be a sklearn or any existed library 
                      object for doing dimension reduction, 
                      you can also write your own dimension reduction 
                      technique for picking specific features or ...,
                      it must have these methods,
                         :fit, for learning a dimension reduction
                         :transform, for transforming given input!
           :verbose: to display some outputs for you!, 
                     (I didn't work on it too much but its neccessary)

        """
        self.transformer = transformer
        self.estimator = estimator
        self.dim_reduction = dim_reduction
        self.verbose = verbose

    def fit(self, X, Y):
        self.transformer.fit(X)
        X_train = self.transformer.transform(X)
        if self.dim_reduction != None:
            if self.verbose:
                print("dimension reduction applied while training your model!")
            self.dim_reduction.fit(X_train)
            X_train = self.dim_reduction.transform(X_train)
        self.estimator.fit(X_train, Y)
    
    def predict(self, X):
        X_test = self.transformer.transform(X)
        if self.dim_reduction != None:
            X_test = self.dim_reduction.transform(X_test)
        return self.estimator.predict(X_test)
    
    def predict_proba(self, X):
        X_test = self.transformer.transform(X)
        return self.estimator.predict_proba(X_test)

