from email import message
from flask import Flask, render_template, request
import pickle
import numpy as np
from senti import text_data_cleaning
from sklearn import utils
#from scipy.sparse import data
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
modelg = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
summarizer = pipeline('summarization') 


model = pickle.load(open('sent.pkl', 'rb'))


app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def hello_world():
    
    #model = load('model.joblib')
    #preds = model.predict(["hello nice to meet you"])
    return render_template('index.html')

# @app.route('/', methods=['POST'])
# def predict():
#     if request.method == "POST":
#        inp = request.form.get("entered")
#        model = load('model.joblib')
#        pred = model.predict(inp)
#     return render_template('index.html',message="pred")

@app.route('/predict', methods=['POST','GET'])
def home():
    inp = request.form['entered']
    if(inp.isnumeric()):
        return render_template('index.html', message="Please enter appropriate sentence")
    arr = np.array([inp])
    pred = model.predict(arr)
    if pred[0] == 1:
#       return render_template('after.html', data=pred)
        return render_template('index.html', message="Positive😄")
    elif pred[0] == 0:
        return render_template('index.html', message="Negative☹️")

@app.route('/predictg', methods=['POST','GET'])
def generate_text():
    
    inpg = request.form['enteredg']
    #arrg = np.array([inpg])
    input_ids = tokenizer.encode(inpg, return_tensors='tf')
    beam_output = modelg.generate(input_ids, max_length=100, num_beans=5, no_repeat_ngram_size=2, early_stopping=True)
    output = tokenizer.decode(beam_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    a =  ".".join(output.split(".")[:-1]) + "."
    return render_template('index.html', message=a)

@app.route('/predicts', methods=['POST','GET'])
def summarize_text():
    
    inps = request.form['entereds']
    text = summarizer(inps, max_lenth=230, min_length=40, do_sample=False)
    sumt = text[0]['summary_text']
    return render_template('index.html', message=sumt)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
  
    text_data_cleaning = text_data_cleaning()
    utils.save_document(text_data_cleaning)
