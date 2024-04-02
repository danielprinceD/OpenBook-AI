from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings 
import torch
import os
import numpy as np
from langchain.chains.question_answering import load_qa_chain
from googletrans import Translator
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from pinecone import Pinecone
from transformers import pipeline
# pc = Pinecone(api_key="0f3718e3-1756-4a37-9583-839c535e296c")
# index_name = "openbook"

# device = torch.device('cpu')
# checkpoint = "MBZUAI/LaMini-T5-738M"
# print(f"Checkpoint path: {checkpoint}")  # Add this line for debugging
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# base_model = AutoModelForSeq2SeqLM.from_pretrained(
#     checkpoint,
#     device_map=device,
#     torch_dtype=torch.float32
# )


def main():
    
    # POST DATA
    
    # loader = PDFMinerLoader('Model/data.pdf')
    # documents = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    # texts = text_splitter.split_documents(documents)
    # text_list = [ texts[i].page_content for i in range(len(texts)) ]
    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # embedded_texts = [ embeddings.embed_query(text) for text in text_list]
    
    # index = pc.Index(index_name)
    # for ids , vect in enumerate(embedded_texts) :
       
    #    index.upsert(vectors=[
    #     {
    #     "id" : str(ids) , 
    #     "values" : vect ,
           
    #     "metadata" : {   
    #         "text" : text_list[ids]
    #     }
    #     }
    # ], namespace = "np1"
    #                  )

    # return
    
    # GET DATA
    
    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # index = pc.Index(index_name)

    # text = "what books written by abdul kalam"
    
    # text_embed = embeddings.embed_query(text)
    
    # get_response = index.query(
    #     namespace = "np4",
    #     vector = text_embed,
    #     top_k =  4,
    #     includeMetadata = True

    # )
    
    # meta = [ i.metadata['text'] for i in  get_response.matches]
    
    
    # pipe = pipeline(
    #     'text2text-generation',
    #     model = base_model,
    #     tokenizer = tokenizer,
    #     max_length = 256,
    #     do_sample = True,
    #     temperature = 0.2,
    #     top_p= 0.20,
    # )
    # local_llm = HuggingFacePipeline(pipeline=pipe)
    
    # chain = load_qa_chain(local_llm , chain_type="stuff")
    # ans = chain.run(input_documents = meta , question = text)
    
    # print(ans)
    translator = Translator()
    print(translator.translate(text = 'what is my name ' , dest = 'ta').text)

    
    

if __name__ == "__main__":
    main()