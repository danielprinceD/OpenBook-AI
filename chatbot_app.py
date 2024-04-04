import streamlit as st 
import os
from googletrans import Translator
import base64
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import torch 
from langchain.chains.question_answering import load_qa_chain
import textwrap 
from langchain.document_loaders import PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.llms import HuggingFacePipeline
from streamlit_chat import message
from pinecone import Pinecone
os.environ['CURL_CA_BUNDLE'] = ''
pc = Pinecone(api_key="0033beeb-4ec5-43e4-87e7-340da2c35638")
index_name = "openbook"
translator = Translator()
st.set_page_config(layout="wide")

device = torch.device('cpu')

checkpoint = "MBZUAI/LaMini-T5-738M"
print(f"Checkpoint path: {checkpoint}")  # Add this line for debugging
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map=device,
    torch_dtype=torch.float32
)



@st.cache_resource
def data_ingestion(filename):
    loader = PDFMinerLoader('Model/'+filename)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    text_list = [ texts[i].page_content for i in range(len(texts)) ]
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    embedded_texts = [ embeddings.embed_query(text) for text in text_list]
    namespace = filename.split("_")[-1].split('.')[0]
    index = pc.Index(index_name)
    for ids , vect in enumerate(embedded_texts) :
       
       index.upsert(vectors=[
        {
        "id" : str(ids) , 
        "values" : vect ,
           
        "metadata" : {   
            "text" : text_list[ids]
        }
        }
    ], namespace = namespace
                     )

    return

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.2,
        top_p= 0.20,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    chain = load_qa_chain(local_llm , chain_type="stuff")
    ans = chain.run(input_documents = meta , question = text)
    return ans



def process_answer(instruction , filename):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    index = pc.Index(index_name)

    text = instruction
    
    text_embed = embeddings.embed_query(text)
    namespace = filename.split("_")[-1].split('.')[0]
    get_response = index.query(
        namespace = namespace,
        vector = text_embed,
        top_k =  4,
        includeMetadata = True

    )
    
    meta = [ i.metadata['text'] for i in  get_response.matches]
    
    
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        top_p= 0.20,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    
    chain = load_qa_chain(local_llm , chain_type="stuff")
    ans = chain.run(input_documents = meta , question = text)
    
    return ans
    

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

@st.cache_data
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    st.markdown(pdf_display, unsafe_allow_html=True)

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))


def main():
 tamil = True
 st.markdown("<h2 style='text-align: center; color:red;'>Upload your PDF</h2>", unsafe_allow_html=True)
 uploaded_file = st.file_uploader("", type=["pdf"])
 
 if uploaded_file is not None :
    filepath = "Model/"+uploaded_file.name
    
    
    
    col1, col2= st.columns([1,2])
    with col1:
        st.markdown("<h4 style color:black;'>Uploaded File</h4>", unsafe_allow_html=True)
        pdf_view = displayPDF(filepath)
    with col2:
        with st.spinner('Embeddings are in process...'):
            ingested_data = data_ingestion(uploaded_file.name)
            st.success('Embeddings are created successfully!')
            
            if st.button("Translate") :
                    tamil = not tamil
            
            st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)


        user_input = st.text_input("", key="input")

        if "generated" not in st.session_state:
            st.session_state["generated"] = ["I am ready to help you"]
        if "past" not in st.session_state:
            st.session_state["past"] = ["Hey there!"]
        
        
        if user_input:
            translated = translator.translate(text=process_answer(user_input , uploaded_file.name) , dest = 'ta').text
            answer = translated if tamil else process_answer(user_input , uploaded_file.name)
            st.session_state["past"].append(user_input)
            response =  answer
            st.session_state["generated"].append(response)
            
            
        if st.session_state["generated"]:
            display_conversation(st.session_state)
        

        




if __name__ == "__main__":
    main()


