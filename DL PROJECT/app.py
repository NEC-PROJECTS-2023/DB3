import streamlit as st
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

st.title('Flower Classifiction')
def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]

    return img_tensor
def save_uploadedfile(uploadedfile):
    tempDir=os.getcwd()
    with open(os.path.join(tempDir,uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())



model = load_model("best_model.epoch25-acc1.00.h5", compile=False)
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0001), metrics=['acc'])

uploaded_files = st.file_uploader("Choose a image", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    st.image(uploaded_file)
       
    save_uploadedfile(uploaded_file)
    img=load_image(uploaded_file.name)
    pred = model.predict(img)
    classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    sci_name=["Bellis perennis","Taraxacum","Rosa","Helianthus","Tulipa"]
    Family=["Asteraceae","Asteraceae","Rosaceae","Asteraceae","Liliaceae"]
   
    pred = pred[:][0].tolist()

    max_prob=max(pred)
    
    x=pred.index(max_prob)
    cls = classes[x]
    sci_n=sci_name[x]
    fam=Family[x]
    print(pred)
    if max_prob>0.83:
        st.write("The predicted class of flower is", cls)
        st.write()
        st.write("The Scientific Name of the flower is ",sci_n)
        st.write()
        st.write("The flower belongs to ",fam,"Family")
    else:
        st.write('your input may not be a flower belonging to','daisy', 'dandelion', 'rose', 'sunflower', 'tulip')
        #st.write('if it was a flower belonging to these classes than it would be',cls)
