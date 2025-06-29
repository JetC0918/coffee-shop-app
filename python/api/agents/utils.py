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
    embeddings = embedding_client.embed_query(text_input) 
    
    return embeddings


def get_embedding_docs(embedding_client, docs_input): 
    embeddings = embedding_client.embed_documents(docs_input) 
    
    return embeddings

def double_check_json_output(client, model_name, json_string):
    prompt = f""" You will check this json string and correct any mistakes that will make it invalid. Then you will return the corrected json string. Nothing else. 
    If the Json is correct just return it.

    Do NOT return a single letter outside of the json string.

    {json_string}
    """
    
    messages = [{'role':'user', 'content': prompt}]
    
    response = get_llm_response(client,model_name, messages)
    return response