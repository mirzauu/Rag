import json
from langchain.text_splitter import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap  = 200,
    length_function = len,
    is_separator_regex = False,
)



def split_data_from_file(file):
		#### define a variable to accumlate chunk records
    chunks_with_metadata = [] 
    
    #### Load json file
    file_as_object = json.load(open(file)) 
    keys = list(file_as_object.keys())
    print(keys)
    #### pull these keys from the json file
    for item in keys: 
        print(f'Processing {item} from {file}') 
        
        #### grab the text of the item
        item_text = file_as_object[item] 
        
        #### split the text into chunks
        item_text_chunks = text_splitter.split_text(item_text) 
        
        chunk_seq_id = 0
        #### loop thtough chunks
        for chunk in item_text_chunks: 
        
		        #### extract file name from each chunk
            form_name = file[file.rindex('/') + 1:file.rindex('.')]
            
            #### create a record with metadata and the chunk text
            chunks_with_metadata.append({
                #### metadata from looping...
                'text': chunk, 
                'Source': item,
                'chunkSeqId': chunk_seq_id,
                #### constructed metadata...
                'chunkId': f'{form_name}-{item}-chunk{chunk_seq_id:04d}',
                
          
            })
            chunk_seq_id += 1
        print(f'\tSplit into {chunk_seq_id} chunks')
    return chunks_with_metadata

"""
This script is designed to process a JSON file by splitting its text content into smaller, structured chunks for easier handling, 
such as for natural language processing or information retrieval tasks. It uses LangChain's RecursiveCharacterTextSplitter to divide 
large text blocks into chunks of up to 2000 characters, with a 200-character overlap to preserve context between segments. 
The function split_data_from_file loads the JSON file, iterates through each key-value pair, and processes the text associated with 
each key. For every piece of text, it creates a series of chunks, each accompanied by metadata including the source key, the order 
of the chunk, and a uniquely formatted chunk ID based on the file name and key name. These enriched chunks are collected into a list 
and returned, making them suitable for tasks like document embedding, indexing, or semantic search."""