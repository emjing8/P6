from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model


        #input shape is the size of the image 150x150
        #categories_count is the number of classes that we have in the dataset

        self.model=Sequential([
            #Rescale pixel values to [0,1]
            Rescaling(1./255, input_shape=input_shape),

            #convolutional layer 
            layers.Conv2D(8, (3,3), activation='relu'),
            layers.MaxPooling2D(3,3), #reduce the size of the image by taking the maximum value in the 2x2 window, cutting the image size in half

            #convolutional layer 2
            layers.Conv2D(16, (3,3), activation='relu'),
            layers.MaxPooling2D(3,3),

            layers.Dropout(0.5),

            #convolutional layer 3
            layers.Conv2D(32, (3,3), activation='relu'),   
            layers.MaxPooling2D(2,2),

            #convolutional layer 4
            layers.Conv2D(64, (3,3), activation='relu'),   
            layers.MaxPooling2D(2,2),


            #Flatten the results to a 1D vecotr
            #layers.GlobalAveragePooling2D(),  #signically lowers params compared to flatten
            layers.Flatten(),
            layers.Dense(32, activation='relu'),

            #dropout layer to prvent overfitting
            layers.Dropout(0.5),

            #output layer
            layers.Dense(categories_count, activation='softmax')
            #softmax activation function is used to output a probability distribution over the classes
        ])


        pass
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup

        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),#optimization algorithm used to minimize the loss function
            loss='categorical_crossentropy', #loss function used for classification problems
            metrics=['accuracy'] #used to monitor the performance of the model
        ), 
        pass