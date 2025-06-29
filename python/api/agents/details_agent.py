from dotenv import load_dotenv
import os
from .utils import get_llm_response, get_embedding
from openai import OpenAI
from copy import deepcopy
from pinecone import Pinecone 
from langchain_huggingface  import HuggingFaceEmbeddings 

load_dotenv()
 
class DetailsAgent():
    def __init__ (self):
        self.client = OpenAI(
            api_key = os.getenv("API_KEY"),
            base_url= os.getenv("BASE_URL")
        )
        
        self.model_name = os.getenv("MODEL_NAME")
    
        self.embedding_client = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device":"cuda"},
            encode_kwargs={"normalize_embeddings": True} 
        )
        
        self.pc = Pinecone(  api_key=os.getenv("PINECONE_API_KEY")  )
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        
    def get_closest_results(self, index_name, input_embeddings, top_k=2):
        index = self.pc.Index(index_name)
        
        results = index.query(
            namespace = 'ns1',
            vector = input_embeddings,
            top_k = top_k,
            include_values = False,
            include_metadata = True
        )
        
        return results

    def get_response(self, messages):
        messages = deepcopy(messages)
        
        user_message = messages[-1]['content'] 
        embeddings = get_embedding(self.embedding_client,  user_message)   
        result = self.get_closest_results(self.index_name, embeddings)
        source_knowledge = "\n".join([x['metadata']['text'].strip() + '\n' for x in result['matches']])
        
        prompt = f"""
        Using the context below, answer the user question:
        Contexts:
        {source_knowledge}
        
        Query: {user_message}
        
        """
        
        system_prompt = """
        You are a heelpful customer support agent for a coffee shop called Sama Sama Coffee.
        You should answer every question as if you are the coffee shop waiter and provide the necessary information to the user regarding their order.
        """
        
        messages[-1]['content'] = prompt
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]
        
        chatbot_output = get_llm_response(self.client, self.model_name, input_messages)
        
        output = self.postprocess(chatbot_output)
        return output

    def postprocess(self, output):
        output = {
                "role": "assistant",
                "content": output ,
                "memory": {
                    'agent' : 'details_agent',   
                } 
        }
        
        return output