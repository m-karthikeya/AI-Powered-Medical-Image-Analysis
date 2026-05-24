#data_loader
import tensorflow as tf

def get_data_generators(data_dir, img_height, img_width, batch_size):
    """Loads and preprocesses images from directories with data augmentation."""
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    print("Loading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    print("Loading Validation Data...")
    val_generator = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    return train_generator, val_generator