'''
Function to visualize Gradient class activation maps
date: 27.07.2019
source: https://github.com/jacobgil/keras-grad-cam
Parameters: -m  : For mobile model
            -t  : For using image from test input
'''


from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2

from keras.models import Model
import os
from keras.models import load_model
import sys

MOBILE = False
nb_classes = 10
if '-m' in sys.argv:
    MOBILE = True
    nb_classes = 5

def checkShape(model):
    s = []
    i = 1
    for layer in model.layers:
        s = layer.output_shape
    if s[1] is 10:
        return("all")
    else:
        return("mobile")

modelPath = 'sampleModel/'
sampleImg = 'c1_img_115.jpg' # training set
sampleImg2 ='c3_img_8.jpg' # test set

def convertIntToClass(n):
    return("c"+str(n))

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = sys.argv[1]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

# block5_conv3 is the last convolution layer
def compile_saliency_function(model, activation_layer='max_pooling2d_3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])
    #print(layer_dict)
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]
        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        #new_model = VGG16(weights='imagenet')
        new_model= loadCorrectmModel()
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x
    
def loadCorrectmModel():
    global MOBILE
    global modelPath
    i = 0
    models = os.listdir(modelPath)
    model = load_model(modelPath + models[i % 2])
    if MOBILE:
        chk = checkShape(model)
        if chk == "all":
            i += 1
            model = load_model(modelPath + models[i % 2])
    else:
        chk = checkShape(model)
        if chk == "mobile":
            i += 1
            model = load_model(modelPath + models[i % 2])
    return(model)

def grad_cam(input_model, image, category_index, layer_name):
    global nb_classes
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(input_model.output)
    model = Model(inputs=input_model.input, outputs=x)
    #model.summary()
    loss = K.sum(model.output)
    conv_output =  [l for l in model.layers if l.name == layer_name][0].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (150, 150))
    #print(output, grads_val)
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    #cam = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

if '-t' in sys.argv:
    img = cv2.imread(sampleImg2)
else:
    img = cv2.imread(sampleImg)
resized_image = cv2.resize(img, (150, 150)) #, target_size=(150, 150))
norm_image = cv2.normalize(resized_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) #input was normalized
normal = np.expand_dims(resized_image, axis=0)
preprocessed_input = np.expand_dims(norm_image, axis=0)


model= loadCorrectmModel()
#model.summary()  # As a reminder.

predictions = model.predict(preprocessed_input)
print(predictions)
predicted_class = np.argmax(predictions)

print('Predicted class:', convertIntToClass(predicted_class))

#Names
if MOBILE:
    nam = "Mobile"+"Gradcam.jpg"
    gnam = "Mobile"+"guided_gradcam.jpg"
else:
    nam = "All"+"Gradcam.jpg"
    gnam = "All"+"guided_gradcam.jpg"

if '-t' in sys.argv:
    nam = "test"+nam
    gnam = "test"+gnam
else:
    nam = "train"+nam
    gnam = "train"+gnam


cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, "max_pooling2d_3")# conv2d_1, max_pooling2d_1 and so on
cv2.imwrite(nam, cam)

register_gradient()
guided_model = modify_backprop(model, 'GuidedBackProp')
print(guided_model.summary())
saliency_fn = compile_saliency_function(guided_model, "max_pooling2d_3") # guided_model: conv2d_1, max_pooling2d_1 and so on
saliency = saliency_fn([preprocessed_input, 0])
gradcam = saliency[0] * heatmap[..., np.newaxis]
cv2.imwrite(gnam, deprocess_image(gradcam))
