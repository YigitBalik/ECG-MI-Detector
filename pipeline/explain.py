import cv2
import os
import math
import glob
import csv
import logging


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from tensorflow import keras
from __main__ import app


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s"
)

# def load_ECGs():
#     dataset = tf.data.Dataset.load(__ECG_PATHS__)
#     X, y = tuple(zip(*dataset))
#     signal, info = tuple(zip(*X))
#     signal, info = np.array(signal), np.array(info)
#     y = np.array(y)
#     return signal, info, y

def explain(image, model, class_index, layer_name, info=None, weighted=None):

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(image, tf.float32)
        conv_outputs, predictions = grad_model((inputs, tf.cast(info.reshape(1,2,1), tf.float32)))
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    guided_grads = (
        tf.cast(conv_outputs > 0, "float32") * tf.cast(grads > 0, "float32") * grads
    )
    output = conv_outputs[0]
    guided_grad = guided_grads[0]
    
    if weighted is not None:
        guided_grad *= weighted.T

    weights = tf.reduce_mean(guided_grad, axis=(0, 1))
    if weights.numpy() == 0:
        weights = 1e-6
    heatmap = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
    
    image = np.squeeze(image)

    heatmap = cv2.resize(heatmap.numpy(), tuple([image.shape[1], image.shape[0]]))
    heatmap = (heatmap - np.min(heatmap)) / (heatmap.max() - heatmap.min())

    return cv2.applyColorMap(
        cv2.cvtColor((heatmap * 255).astype("uint8"), cv2.COLOR_GRAY2BGR), cv2.COLORMAP_JET
    )

def plot_ecg_image(ax, sensor_data, lead, heatmap, name):

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False) 
    ax.spines["left"].set_visible(False) 
    ax.spines["bottom"].set_visible(False) 

    ax.tick_params(axis=u'both', which=u'both',length=0)

    heatmap = heatmap / 255

    data_points = np.zeros((len(sensor_data), 1, 2))

    for row_index, point in enumerate(sensor_data):
        data_points[ row_index, 0, 0 ] = row_index
        data_points[ row_index, 0, 1 ] = point

    segments = np.hstack([data_points[:-1], data_points[1:]])
    coll = LineCollection(segments, colors=[ [ 0, 0, 0 ] ] * len(segments), linewidths=(1.3)) 
    
    ax.add_collection(coll)
    ax.autoscale_view()

    # colors = np.mean(heatmap, axis=0)
    colors = heatmap[lead]
    for c_index, color in enumerate(colors):
        ax.axvspan(c_index, c_index+1, facecolor=color)

def visualize_ecg_prediction_tf_explain(model_path, samples, info, preds, output_path, class_indicies=[ 0, 1 ]):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    
    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception:
        logging.error("Could not load %s!" % model_path)
        return
    
    if not os.path.exists(output_path):
            os.makedirs(output_path)
    
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    layer_names = [ layer.name for layer in model.layers if type(layer) == keras.layers.Conv1D ][ -1 : ]
    
    logging.info("Starting to visualize the predictions of %i ECGs" % len(samples))

    for ecg_index, sample in enumerate(samples):
        
        logging.info("Visualizing %s..." % ecg_index)
        ecg_values = np.array(sample, dtype=np.float32)
        ecg_values = np.reshape(ecg_values, (1, *ecg_values.shape))
        
    
        col = "Activation Map"
        rows = [ "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        ecg_name = ecg_index
        
        plt.figure()
        plt.axis("off")

        fig, axes = plt.subplots(len(rows), 1, figsize=(10, 18))
        
        pred = np.argmax(preds[ecg_index])
        ratio = preds[ecg_index][pred]
        
        # ax.set_title(col + "\nprediction: " + str(pred) + " ratio: " + str(ratio), fontsize= 9)
        fig.suptitle(col + "\nprediction: " + str(pred) + " ratio: " + str(ratio), fontsize=20)

        for index, (ax, row) in enumerate(zip(axes, rows)):
            ax.set_ylabel(row, fontsize=20)

        # ax.set_ylabel(rows[0], fontsize=9)
        
        for layer_index, layer_name in enumerate(layer_names):

            logging.info("Visualizing layer %s..." % layer_name)

            output = explain(
                ecg_values,
                model=model,
                layer_name=layer_name,
                class_index=pred,
                info=info[ecg_index]
            )

            output = cv2.cvtColor(np.transpose(output, (1, 0, 2)), cv2.COLOR_BGR2RGB)
            for lead in range(len(rows)):
                sensor_values = np.transpose(ecg_values.squeeze())[lead]
                plot_ecg_image(axes[lead], sensor_values, lead, output, "%s_%s" % (ecg_name, layer_name))

        file_format = "png"
        image_output_name = "%s_%s.%s" % (model_name, ecg_name, file_format)
    
        logging.info("Saving %s..." % image_output_name)

        plt.savefig(
            fname=os.path.join(output_path, image_output_name),
            format=file_format,
            dpi=600,
        )


        # fig.savefig(fname=os.path.join(output_path, "last_conv_" + image_output_name), bbox_inches=axes[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted()))

        plt.cla()
        plt.clf()
        plt.close("all")
    
def interpret(folder, ecg, info, results):
    print("SHAPE", ecg.shape, info.shape)
    visualize_ecg_prediction_tf_explain(os.path.join(app.config["MODEL_PATH"], "best_model.hdf5"), 
                                        ecg,
                                        info, 
                                        results,  
                                        "./static/", class_indicies=[0,1])

