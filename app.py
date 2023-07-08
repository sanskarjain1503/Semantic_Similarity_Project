import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

model=pickle.load(open('models/model.pkl','rb'))

@app.route('/',methods=['GET','POST'])
def semantic_similarity1():
    if request.method=='POST':
        text1=request.form.get('text1')
        text2=request.form.get('text2')
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
        results=cosine_scores.item()
        if results > 0.5:
            results=1
        else:
            results=0
        return render_template('home.html',result=results,text1=text1,text2=text2)
        
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")