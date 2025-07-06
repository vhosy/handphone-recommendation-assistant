from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS 
from langchain_core.documents import Document
import pandas as pd
import json

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def create_vector_store (df, embeddings):
    df_dict = df.to_dict(orient='records')
    df_json = [json.dumps(x) for x in df_dict]
    doc = [Document(page_content = x) for x in df_json]
    
    vector_store = FAISS.from_documents(doc, embedding=embeddings)
    
    return vector_store

handsets_df = pd.read_excel('data/handsets_db.xlsx')
vector_store = create_vector_store (handsets_df, embeddings)
vector_store.save_local("vectorstore", "handsets")      

customers_df = pd.read_excel('data/customer_db.xlsx') 
vector_store = create_vector_store (customers_df, embeddings)
vector_store.save_local("vectorstore", "customers")            
