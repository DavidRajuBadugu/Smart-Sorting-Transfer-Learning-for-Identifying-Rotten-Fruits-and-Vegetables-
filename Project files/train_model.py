from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
trainpath = "dataset/train"
testpath = "dataset/test"
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, shear_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory(trainpath, target_size=(224, 224), batch_size=20)
test = test_datagen.flow_from_directory(testpath, target_size=(224, 224), batch_size=20)
vgg = VGG16(include_top=False, input_shape=(224, 224, 3))

# Freeze all VGG layers
for layer in vgg.layers:
    layer.trainable = False

# Add new layers
x = Flatten()(vgg.output)
output = Dense(28, activation='softmax')(x)  # For 28 classes

model = Model(inputs=vgg.input, outputs=output)

model.summary()
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
opt = Adam(learning_rate=0.0001)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train,
    validation_data=test,
    epochs=25,
    steps_per_epoch=20,
    callbacks=[early_stopping]
)
model.save('healthy_vs_rotten.h5')  # Saves the model as a folder
