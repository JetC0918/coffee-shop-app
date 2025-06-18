def  get_llm_response(client,model_name, messages, temperature = 0):
    input_messages = []
    for message in messages:
        input_messages.append({"role": message['role'], "content": message['content']})
    
    response = client.chat.completions.create(
        model= model_name,
        messages= input_messages,
        temperature = temperature, 
    ).choices[0].message.content
    
    return response
 

def get_embedding(embedding_client, text_input): 
    embeddings = embedding_client.embed_documents(text_input) 
    
    return embeddings


def get_embedding_docs(embedding_client, docs_input): 
    embeddings = embedding_client.embed_docs(docs_input) 
    
    return embeddings