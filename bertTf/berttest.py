# Import libraries
import os
import tensorflow as tf
#import tensorflow_models as tfm
import official.nlp.modeling.models as tfm

import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Load the BERT model from TF Hub
bert_model_name = 'bert_en_uncased_L-12_H-768_A-12'
tfhub_handle_encoder = 'd:/Tensorflowhub_model/bert_en_uncased_L-12_H-768_A-124/4'
tfhub_handle_preprocess = 'd:/Tensorflowhub_model/bert_en_uncased_L-12_H-768_A-124/preprocess'

bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
bert_model = hub.KerasLayer(tfhub_handle_encoder)

# Load the GLUE MRPC dataset from TFDS
batch_size = 32
glue, info = tfds.load('glue/mrpc', with_info=True, batch_size=batch_size)
train_dataset = glue['train']
validation_dataset = glue['validation']
test_dataset = glue['test']

# Modify your dataset to use 'text' as input
# Modify your dataset to use 'text' as input and 'label' as target
train_dataset = train_dataset.map(lambda x: ({'text': x['sentence1'] + ' ' + x['sentence2']}, x['label']), num_parallel_calls=tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.map(lambda x: ({'text': x['sentence1'] + ' ' + x['sentence2']}, x['label']), num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.map(lambda x: ({'text': x['sentence1'] + ' ' + x['sentence2']}, x['label']), num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Define the model architecture
def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = bert_preprocess_model(text_input)
  encoder_inputs = preprocessing_layer
  encoder = bert_model(encoder_inputs)
  outputs = encoder['pooled_output']
  net = outputs
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)

# Create the classifier model
classifier_model = build_classifier_model()

# Compile the model with loss and metrics
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()
epochs = 5
steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)
init_lr = 3e-5
from official.nlp.optimization import create_optimizer
optimizer = create_optimizer(init_lr=init_lr,
                              num_train_steps=num_train_steps,
                              num_warmup_steps=num_warmup_steps,
                              optimizer_type='adamw')




classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

# Train the model
history = classifier_model.fit(x=train_dataset,
                               validation_data=validation_dataset,
                               epochs=epochs)

# Evaluate the model on the test set
loss, accuracy = classifier_model.evaluate(test_dataset)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
