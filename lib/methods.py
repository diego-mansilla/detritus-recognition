import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn import metrics

def test_accuracy(model, test_dataset):
    loss, accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)
    return accuracy

def train_model(model, epochs, train_dataset, validation_dataset, lr, callbacks):
    print("Training model, epochs: ", epochs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    history = model.fit(train_dataset,
                    epochs=epochs,
                    validation_data=validation_dataset,
                    callbacks=[callbacks])
    
    return history


def print_tsne(model, dataset, n_iter = 1000):
    new_ds = dataset
    
    x_dataset = []
    y_dataset = []
    
    for x, y in new_ds:
        x_dataset.append(x)
        y_dataset.append(y)
    
    small_dataset = np.concatenate(x_dataset)
    y_dataset = np.concatenate(y_dataset)
    
    results = model.predict(small_dataset)
    scores = tf.nn.sigmoid(results)
    labels = tf.where(scores < 0.5, 0, 1)
    
    colors = ['red', 'blue']
    classes = dataset.class_names
    
    model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    features = model2.predict(small_dataset)

    tsne = TSNE(n_components=2, verbose=1, perplexity=50
                , n_iter=n_iter)
    tsne_results = tsne.fit_transform(features)
    
    tx = tsne_results[:, 0]
    ty = tsne_results[:, 1]
    
    fig2 = plt.figure(figsize=(16,16))
    fig2.suptitle('TSNE with prediction labels', fontsize=20)
    ax2 = fig2.add_subplot(111)
    for idx, c in enumerate(colors):
        indices = [i for i, l in enumerate(labels) if idx == l]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        ax2.scatter(current_tx, current_ty, c=c, label=classes[idx])
    
    ax2.legend(loc='best')
   
    fig = plt.figure(figsize=(16,16))
    fig.suptitle('TSNE with Ground Truth labels', fontsize=20)
    ax = fig.add_subplot(111)
    for idx, c in enumerate(colors):
        indices = [i for i, l in enumerate(y_dataset) if idx == l]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        ax.scatter(current_tx, current_ty, c=c, label=classes[idx])
    
    ax.legend(loc='best')

def show_plot(history, drop_value=0.0):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title("Training and Validation Loss with Accuracy {0}".format(drop_value))

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title("Training and Validation Loss with Dropout {0}".format(drop_value))
    plt.xlabel('epoch')
    plt.show()

def show_report(model, generator):
    test_steps_per_epoch = np.math.ceil(generator.samples / generator.batch_size)
    predictions = densenet_model.predict(generator, steps=test_steps_per_epoch)
    
    y_pred = np.empty(len(predictions), dtype=float) 
    for i in range(len(predictions)):
        score = tf.nn.sigmoid(predictions[i])
        y_pred[i] = tf.where(score < 0.5, 0, 1)
    
    true_classes = generator.classes
    class_labels = list(generator.class_indices.keys())   
    report = metrics.classification_report(true_classes, y_pred, target_names=class_labels)
    print(report)
    
def show_confusion_matrix(model, generator):
    test_steps_per_epoch = np.math.ceil(generator.samples / generator.batch_size)
    predictions = model.predict(generator, steps=test_steps_per_epoch)
    
    y_pred = np.empty(len(predictions), dtype=float) 
    for i in range(len(predictions)):
        score = tf.nn.sigmoid(predictions[i])
        y_pred[i] = tf.where(score < 0.5, 0, 1)
    
    detr_list = ['LClass_Detritus', 'LClass_Bubbles', 'LClass_shadow']
    
    true_classes = generator.classes
    filepaths = generator.filepaths
    label_map = generator.class_indices
    index_map = {v: k for k, v in label_map.items()}
    
    class_labels = list(generator.class_indices.keys()) 
    correct =  {new_key: 0 for new_key in class_labels}
    incorrect = {new_key: 0 for new_key in class_labels}
    correct_files = []
    incorrect_files = []
    
    
    for i in range(len(true_classes)):
        if any(index_map[true_classes[i]] in s for s in detr_list):
            if (y_pred[i] == 0):
                correct[index_map[true_classes[i]]] += 1
                correct_files.append(filepaths[i])
            else:
                incorrect[index_map[true_classes[i]]] += 1
                incorrect_files.append(filepaths[i])
        else:
            if (y_pred[i] == 1):
                correct[index_map[true_classes[i]]] += 1
                correct_files.append(filepaths[i])
            else:
                incorrect[index_map[true_classes[i]]] += 1
                incorrect_files.append(filepaths[i])
    # report = metrics.classification_report(true_classes, y_pred, target_names=class_labels)
    for class_name in class_labels:
        corr = correct[class_name]
        incorr = incorrect[class_name]
        total = corr + incorr
        if (total == 0):
            corr_perc = 0
        else:
            corr_perc = corr/total
        print(class_name, corr, incorr, total, corr_perc)
    return correct, incorrect, correct_files, incorrect_files