!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

from fastbook import *
from fastai.vision.widgets import *

path = untar_data(URLs.PETS)/'images'

def is_dog(x): return x[0].islower()

dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_dog, item_tfms=Resize(224))

learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(2)

import cloudpickle

learn.export(fname='model.pkl', pickle_module=cloudpickle)
output = load_learner(path/'model.pkl')

from fastai.vision.widgets import *
from fastai.vision.all import *

from pathlib import Path

import streamlit as st

class Predict:
    def __init__(self, filename):
        self.learn_inference = load_learner(path/filename)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
            self.get_prediction()
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(500,500), caption='Imagem para análise')

    def get_prediction(self):

        if st.button('Classificar'):
            is_dog,_,probs = learn.predict(img)

            if str(is_dog) == 'True': 
              resposta = "cachorro"
              probabilidade = probs[1].item() * 100

            else: 
              resposta = "gato"
              probabilidade = 100 - (probs[1].item()*100)

              st.write(f"É um: {resposta}.")
              st.write(f"A chance de ser um {resposta} é: {probabilidade:.2f}%")

        else: 
            st.write(f'Clique para classificar') 

if __name__=='__main__':

    file_name='model.pkl'

    predictor = Predict(file_name)
