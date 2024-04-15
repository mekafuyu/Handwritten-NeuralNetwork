import os
import tensorflow as tf
from keras import models, layers, activations,\
optimizers, utils, losses, initializers, metrics, callbacks

class DefaultModel:
  model = None
  train = None
  test = None
  
  def __init__(self, model_path, train_path, epochs, batch_size, patience, learning_rate) -> None:
    self._epochs = epochs
    self._batch_size = batch_size
    self._patience = patience
    self._learning_rate = learning_rate
    
    self._model_path = model_path
    self._train_path = train_path
    
    self._exists = os.path.exists(model_path)
    self._items = len(os.listdir(self._train_path))
    self._classes = 1 if self._items < 3 else self._items
    print(f"Model with {self._classes} classes")
  
  def CreateLoadModel(self):
    # Carrega modelo se já existir um checkpoint, caso contrário, o cria.
    if self._exists:
      self.model = models.load_model(self._model_path)
    else:
      self.model = models.Sequential([
        layers.Resizing(128, 128),
        layers.Rescaling(1.0/255),
        layers.Conv2D(62, (3, 3),
          activation = 'relu',
          kernel_initializer = initializers.RandomNormal()
        ),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(124, (3, 3),
          activation = 'relu',
          kernel_initializer = initializers.RandomNormal()
        ),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(248,
          activation = 'relu',
          kernel_initializer = initializers.RandomNormal()
        ),
        layers.Dropout(0.5),
        layers.Dense(124,
          activation = 'relu',
          kernel_initializer = initializers.RandomNormal()
        ),
        layers.Dense(62,
          activation = 'sigmoid',
          kernel_initializer = initializers.RandomNormal()
        )
        ])
    
    if not self._exists:
      self.model.compile(
        optimizer = optimizers.Adam(
          learning_rate = self._learning_rate
        ),
        loss = losses.SparseCategoricalCrossentropy(),
        metrics = [ 'accuracy' ]
      )
    else:
      self.model.summary()
      
  def Split(self):
    self.train = utils.image_dataset_from_directory(
      self._train_path,
      validation_split= 0.2,
      subset= "training",
      seed= 123,
      shuffle= True,
      image_size= (128, 128),
      batch_size= self._batch_size
    )
    self.test = utils.image_dataset_from_directory(
      self._train_path,
      validation_split= 0.2,
      subset= "validation",
      seed= 123,
      shuffle= True,
      image_size= (128, 128),
      batch_size= self._batch_size
    )

  def Fit(self):
    # lr_schedule = optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-4,
    #     decay_steps=100,
    #     decay_rate=0.5)

    self.model.fit(self.train,
      epochs = self._epochs,
      validation_data = self.test,
      callbacks= [
        callbacks.EarlyStopping(
          monitor = 'val_loss',
          patience = self._patience,
          verbose = 1
        ),
        callbacks.ModelCheckpoint(
          filepath = self._model_path,
          save_weights_only = False,
          monitor = 'loss',
          mode = 'min',
          save_best_only = True
        ),
        # callbacks.LearningRateScheduler(
        #   lr_schedule 
        # )
      ]
    )