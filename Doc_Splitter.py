from langchain_community.document_loaders import DirectoryLoader
from langchain.schema.document import Document
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import os
from dotenv import load_dotenv

device= 'cuda' if torch.cuda.is_available() else 'cpu'

load_dotenv()
API_KEY = os.getenv('API_KEY')

pc = Pinecone(api_key=API_KEY)

DATA_PATH = r''
DIMENSIONS= 1024
METRIC= 'cosine'
TOP_K = 3


# Initialize tokenizer and model
model_path = 'Alibaba-NLP/gte-large-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)


def main():

    # Create (or update) the data store.
    documents = load_documents(DATA_PATH)
    print(documents[0])     
    # document =  (page_content(string), metadata : {"source" :" " })

def load_documents(Data_Path= DATA_PATH):
    """
    Loads all text files from a folder into list of Document type
    """
    loader = DirectoryLoader(Data_Path, glob="*.txt")
    docs = loader.load()
    print(len(docs), "documents loaded")
    return docs

def generate_ids(documents: list[Document]):
    """
    Updates metadata of documents with key id.
    Values of ids are automatically generated in a custom format
    """
    for doc in documents:
        source_path = doc.metadata.get('source', '')
        if source_path:
            file_name = os.path.basename(source_path)
            doc_id = file_name[:10]
            doc.metadata['id'] = doc_id

def create_index(index_name : str, dimensions= 1024):
    """
    Creates a new index in the pinecone database if given name is not already found\n
    NOTE: DONOT USE _ IN NAME. ONLY ALPHANUMERIC CHARACTERS AND '-' ARE ALLOWED
    """
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name= index_name,
            dimension= dimensions,
            metric= "cosine",
            spec= ServerlessSpec(
                cloud= 'aws',
                region= 'us-east-1'
            )
        )
    index = pc.Index(index_name)
    return index

def add_to_index(index_name : str, documents: list[Document]):
    """
    Embeds Document and adds it as a record\n
    NOTE: This function overwrites record if id exists already.\n
    To update metadata instead, use update_metadata() function
    """
    # Collect embeddings
    list_page_contents = [doc.page_content for doc in documents]
    embeddings = embed_docs(list_page_contents)

    # Generate ids
    generate_ids(documents)

    # Fill the vectors in the required format
    vectors= []
    for doc, embedding in zip(documents, embeddings):
        vectors.append({
            "id": doc.metadata['id'],
            "values": embedding,
            "metadata": {**{key: value for key, value in doc.metadata.items() if key != 'id'},
                         'text': doc.page_content}
        })
    # Upload vectors
    index = pc.Index(index_name)
    index.upsert(
        vectors=vectors,
        namespace="Internships"
    )
    print (len(vectors), 'items added')

def update_metadata(index_name : str, namespace : str, record_id : str, metadata : dict):
    """
    Update metadata of the record without modifying the vector.\n
    If key exists already, value is modified\n
    Else, new key-value pair is added.
    """
    index = pc.Index(index_name)
    index.update(
        id = record_id,
        set_metadata= metadata,
        namespace= namespace
    )

def retreive_data(index_name : str, namespace : str, query : str, metadata_filter : dict):
    """
    Similarity search for given query with applied filter.\n
    Returns results along with metadata
    """
    # Embed query text
    query_vector = embed_docs([query])[0]
    
    index = pc.Index(index_name)
    
    # Retreive results with applied filter 
    results = index.query(
        namespace= namespace,
        vector= query_vector,
        filter= filter_format(metadata_filter),
        top_k= TOP_K,
        include_metadata= True
    )
    return results

def filter_format(metadata_filter: dict) -> dict:
    """
    Converts filters into the format as required by Pinecone
    """
    # Create a list of filter conditions
    filter_conditions = []
    
    for key, value in metadata_filter.items():
        # Add each key-value pair as a filter condition
        filter_conditions.append({key: {"$eq": value}})
    
    # Create the final filter dictionary with $and operator
    filter_dict = {"$and": filter_conditions}
    
    return filter_dict

def embed_docs(docs: list[str]):
    """
    Embedding function.\n 
    Returns a vector of particular dimension based on model used\n
    Not for external call.
    """

    # tokenize
    tokens = tokenizer(
            docs, padding=True, max_length=512, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        # process with model for token-level embeddings
        out = model(**tokens)
        # mask padding tokens
        last_hidden = out.last_hidden_state.masked_fill(
                ~tokens["attention_mask"][..., None].bool(), 0.0
        )
        # create mean pooled embeddings
        doc_embeds = last_hidden.sum(dim=1) / tokens["attention_mask"].sum(dim=1)[..., None]
    return doc_embeds.cpu().numpy().tolist()

if __name__ == "__main__":
    main()
