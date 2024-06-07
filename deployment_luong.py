# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:13:50 2024

@author: ADMIN
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:41:32 2024

@author: ADMIN
"""

import streamlit as st
import pickle
import pandas as pd
import os


def predict(namkinhnghiem):    # load mô hình
    loaded_model = pickle.load(open(r'.\model\model_luong.sav','rb'))
    row=[]
    row.append(namkinhnghiem)

    data=[]
    data.append(row)
    df=pd.DataFrame(data)
    y_pred=loaded_model.predict(df)
    return y_pred

st.title("Ứng dụng Streamlit dự đoán")
namkinhnghiem = st.number_input("Nhập số năm kinh nghiệm",value=2)

if st.button('Dự đoán'):
    output_value = predict(namkinhnghiem)
    
    
    st.write("Giá trị đầu ra dự đoán:", output_value)