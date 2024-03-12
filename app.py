import base64
import streamlit as st
import torch
from pptx import Presentation
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import pipeline
from pdf2image import convert_from_path
import os
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

# MODEL and TOKENIZER
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)


# Function to convert PPTX or PPT to PDF using unoconv
def convert_ppt_to_pdf(input_file_path, output_file_prefix):
  output_file_path = f"{output_file_prefix}.pdf"
  command = f"unoconv -f pdf -o {output_file_path} {input_file_path}"
  os.system(command)
  return output_file_path


# Display function for PDF using images
def display_pdf(file_path):
  images = convert_from_path(file_path)
  for image in images:
    st.image(image)


st.title("PPT/PPTX Viewer")

uploaded_file = st.file_uploader("Choose a PPT or PPTX file", type=["ppt", "pptx"])
if uploaded_file is not None:
  # Save uploaded file to disk with appropriate extension
  file_extension = uploaded_file.name.split('.')[-1]
  temp_file_path = f"temp_presentation.{file_extension}"
  with open(temp_file_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

  # Convert PPT or PPTX to PDF
  pdf_path = convert_ppt_to_pdf(temp_file_path, "output_pdf")

  # Display the PDF
  display_pdf(pdf_path)

  # Cleanup
  os.remove(temp_file_path)
  os.remove(pdf_path)


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
  input_pdf = file_processing(filepath)
  result = pipeline_summarizer(input_pdf)
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
  st.title('Online PPTX Summarizer')
  uploaded_file = st.file_uploader("Upload your PPTX File", type=['pptx'])

  if uploaded_file is not None:
    if st.button("Summarize"):
      col1, col2 = st.columns(2)
      filepath = "data/"+uploaded_file.name
      with open(filepath, 'wb') as temp_file:
        temp_file.write(uploaded_file.read())

      with col1:
        st.info("Uploaded PDF File")
        pdf_viewer = displayPPTX(filepath)


      with col2:
        st.info("Here is your PDF Summarization")
        summary = LLM_Pipeline(filepath)
        st.success(summary)


if __name__ == "__main__":
  main()