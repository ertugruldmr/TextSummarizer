import gradio as gr
import pickle
import pickle
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, pipeline

# File Paths
model_path = 'fine_tuned_sum' 
tokenizer_path = "tokenizer"
examples_path = "examples.pkl"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Load the fine-tuned BERT model
seq2seq_model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)

# loading the examples
with open('examples.pkl', 'rb') as f: examples = pickle.load(f)

# Creating the pipeline
sum_params = {
    "model":seq2seq_model,
    "tokenizer":tokenizer,
    "framework":"tf",
}

summarizer = pipeline("summarization", **sum_params)
# Load the model
# Define a function to make predictions with the model
def summarize(text):

    # defining the params
    prms = {
        "min_length":5,
        "max_length":128
    }
    return summarizer(text,**prms)[0]["summary_text"]

# GUI Component
# defining the params
if_p = {
    "fn":summarize,
    "inputs":gr.inputs.Textbox(label="Text"),
    "outputs":gr.outputs.Textbox(label="Output"),
    "title":"Fine-tuned 't5-small' model for text summarization",
    "description":"Write something to summarization text",
    "examples":examples
}

# Create a Gradio interface instance
demo = gr.Interface(**if_p)

# Launching the demo
if __name__ == "__main__":
    demo.launch()
