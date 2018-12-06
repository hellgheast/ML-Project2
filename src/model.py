from data import *
from unet import unet
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

def train_model():
    X_train, Y_train = load_train_set()
    model = unet()

    val_split = 0.1
    idx_x = int(X_train.shape[0]*(1.0-val_split))
    idx_y = int(Y_train.shape[0]*(1.0-val_split))
    X_TR  = X_train[:idx_x]
    Y_TR  = Y_train[:idx_y]
    X_VAL = X_train[idx_x:]
    Y_VAL = Y_train[idx_y:]

    datagen_dict = dict(featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=45,
        vertical_flip=True,
        horizontal_flip=True,
        zoom_range = 0.1,
        width_shift_range = 0.1,
        height_shift_range = 0.1,                
        fill_mode='reflect',
        dtype=np.uint8)

    datagenx_tr = ImageDataGenerator(**datagen_dict)
    datageny_tr = ImageDataGenerator(**datagen_dict)

    x_tr_gr  = datagenx_tr.flow(X_TR, batch_size=1,seed=1)
    y_tr_gr  = datageny_tr.flow(Y_TR, batch_size=1,seed=1)

    tr_gr  = zip(x_tr_gr,y_tr_gr)
        
    earlystopper = EarlyStopping(patience=10, verbose=1)
    checkpointer = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
    results = model.fit_generator(tr_gr,
                    steps_per_epoch= 200,#len(X_TR),
                    epochs=50,
                    validation_data=(X_VAL, Y_VAL),
                    validation_steps=len(X_VAL),          
                    callbacks=[earlystopper, checkpointer])

def predict():
    X_test = downscale_test_images(load_test_set())
    model = load_model('model.h5')
    
    predictions = model.predict(X_test, verbose=1)
    predictions = (predictions > 0.5).astype(np.uint8)
    
    imgs = upscale_predictions(predictions)

    for i in range(len(imgs)):
        imsave("results/test_" + str(i + 1) + ".png", imgs[i]) #*255
