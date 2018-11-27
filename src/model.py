from data import *
from unet import unet
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

def train_model():
    X_train, Y_train = load_train_set()
    model = unet()

    earlystopper = EarlyStopping(patience=10, verbose=1)
    checkpointer = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
    model.fit(X_train, Y_train, validation_split=0.1, batch_size=2, epochs=50, callbacks=[earlystopper, checkpointer])

def predict():
    X_test = downscale_test_images(load_test_set())
    model = load_model('model.h5')
    
    predictions = model.predict(X_test, verbose=1)
    predictions = (predictions > 0.5).astype(np.uint8)
    
    imgs = upscale_predictions(predictions)

    for i in range(len(imgs)):
        imsave("results/test_" + str(i + 1) + ".png", imgs[i]) #*255
