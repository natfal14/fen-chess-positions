import streamlit as st
import numpy as np
import cv2
import joblib

st.sidebar.title("AI Chess Master")
st.write("Welcome to AI Chess Master. Upload an image of a chessboard to get its FEN positions. To get images, download them from this link https://www.kaggle.com/datasets/koryakinp/chess-positions. Make sure you select an image from the folder dataset/test/")

uploaded_file = st.sidebar.file_uploader("Upload an image:")
model = joblib.load('model2.joblib')

def get_pred(im):
    
    classes = ['P', 'K', 'Q', 'B', 'N', 'R', 'p', 'k', 'q', 'b', 'n', 'r', 'Empty']
    
    prediction = model.predict(im)
    prediction = prediction[0].tolist()
    max_pred = max(prediction)
    max_index = prediction.index(max_pred)
    pred = classes[max_index]
    print(prediction)
    
    return pred

def get_board_labeled(im_path):
        
    im = cv2.imread(im_path)
    label = ''
    preds = []
    empty = 0
    flag = 0
    for i in range(8):
        for j in range(8):
            cropped_im = im[0+50*i:50+50*i, 0+50*j:50+50*j]
            cropped_im = cropped_im.astype('float32')
            cropped_im = np.expand_dims(cropped_im, axis=0)
            new_pred = get_pred(cropped_im)      
            
            if new_pred == 'Empty':
                flag = 1
                empty += 1
                new_pred = ''
                    
            else:
                flag = 0
                
            if (empty > 0 and flag == 0) or (empty > 0 and j == 7):
                label += str(empty)
                label += new_pred
                empty = 0

            else:
                label += new_pred

        label = label + '-'
        
    return label[0:-1]

    
if uploaded_file is not None:
    
    path = 'dataset/test/' + uploaded_file.name
    label = get_board_labeled(path)
    st.write('Predicted string: ' + label)
    st.write('Actual string: ' + uploaded_file.name[:-5])
    st.image(uploaded_file)