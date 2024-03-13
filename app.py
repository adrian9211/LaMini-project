import base64
import streamlit as st
import torch
import streamlit_shadcn_ui as ui
from pptx import Presentation
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import pipeline


# MODEL and TOKENIZER
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

#File Loader and Preprocessing
def file_processing(file):
  presentation = Presentation(file)
  text = []
  for slide in presentation.slides:
    for shape in slide.shapes:
      if hasattr(shape, "text"):
        text.append(shape.text)
  return "\n".join(text)

def extract_text_from_ppt(ppt_file):
  presentation = Presentation(ppt_file)
  text = []
  for slide in presentation.slides:
    for shape in slide.shapes:
      if hasattr(shape, "text"):
        text.append(shape.text)
  return "\n".join(text)

#LLM pipeline
def LLM_Pipeline(filepath):
  pipeline_summarizer = pipeline(
    'summarization',
    model = base_model,
    tokenizer = tokenizer,
    max_length = 500,
    min_length = 25
  )
  input_pptx = file_processing(filepath)
  result = pipeline_summarizer(input_pptx)
  result = result[0]['summary_text']
  return result

@st.cache_data
#function to display the pdf of a given file
def displayPPTX(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PPTX in HTML
    pptx_display = F'<embed src="data:application/pptx;base64,{base64_pdf}" width="700" height="1000" type="application/pptx">'

    # Displaying File
    st.markdown(pptx_display, unsafe_allow_html=True)

#Streamlit code
# st.set_page_config(layout='wide', page_title="PPTX Summarizer")

def main():
  st.title('Legal-Pythia')
  st.subheader('Online PPTX Summarizer')
  uploaded_file = st.file_uploader("Upload your PPTX File", type=['pptx'])

  if uploaded_file is not None:
    if st.button("Summarize"):

      filepath = "data/"+uploaded_file.name
      with open(filepath, 'wb') as temp_file:
        temp_file.write(uploaded_file.read())


        st.info("Here is your PDF Summarization")
        summary = LLM_Pipeline(filepath)
        st.success(summary)

if __name__ == "__main__":
  main()