import tensorflow as tf
# from tensorflow import Keras 
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
st.header('Image Classification Model')

data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']
model = load_model('Image_classification_vegitable_fruit.keras')
img = st.text_input("Enter image name","image_2.jpg")
img_height=180
img_width=180

image = tf.keras.utils.load_img(img, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image)
img_bat = tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image,width = 200)
st.write(f'Vag/Fruit in image is {data_cat[np.argmax(score)]} with accuracy of {np.max(score*100):0.2f}')