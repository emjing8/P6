from keras.utils import image_dataset_from_directory
from config import train_directory, test_directory, image_size, batch_size, validation_split

#traing set: used for learning patterns in the data
#validation set:used to monitor performance durring training and to decide when to stop training.
#test set: used after training and to evaluate the final model perofrmance

def _split_data(train_directory, test_directory, batch_size, validation_split):
    print('train dataset:')
    train_dataset, validation_dataset = image_dataset_from_directory(
        train_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split, #split the data into training and validation, the validation well be used to make sure the learning network on unseen data
        #as its in the learing proccess
        subset="both",
        seed=47
    )
    print('test dataset:')
    test_dataset = image_dataset_from_directory(
        test_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )

    return train_dataset, validation_dataset, test_dataset

def get_datasets():
    train_dataset, validation_dataset, test_dataset = \
        _split_data(train_directory, test_directory, batch_size, validation_split)
    return train_dataset, validation_dataset, test_dataset

def get_transfer_datasets():
    # Your code replaces this by loading the dataset
    # you can use image_dataset_from_directory, similar to how the _split_data function is using it
    train_dataset, validation_dataset, test_dataset = None, None, None
    # ...
    return train_dataset, validation_dataset, test_dataset