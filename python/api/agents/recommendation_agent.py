from openai import OpenAI
import os
import pandas as pd
import json
from copy import deepcopy
from .utils import get_llm_response, double_check_json_output
from dotenv import load_dotenv
from typing import List, Dict, Any
load_dotenv() 


class RecommendationAgent():
    def __init__ (self, apriori_recommendation_path, popular_recommendation_path): 
        self.client = OpenAI(
            api_key = os.getenv("API_KEY"),
            base_url= os.getenv("BASE_URL")
        )
        
        self.model_name = os.getenv("MODEL_NAME")
        
        with open(apriori_recommendation_path,'r') as file:
            self.apriori_recommendations = json.load(file)
            
        self.popular_recommendations = pd.read_csv(popular_recommendation_path)
        
        self.products = self.popular_recommendations['product'].tolist()
        self.product_categories = list(set(self.popular_recommendations['product_category'].tolist()))
        
    def get_popular_recommendation(self, product_categories=None, top_k=5):
        recommendations_df = self.popular_recommendations.copy()
         
        if type(product_categories) == str:
            product_categories = [product_categories]

        if product_categories is not None:
            recommendations_df = self.popular_recommendations[self.popular_recommendations['product_category'].isin(product_categories)]
        recommendations_df = recommendations_df.sort_values(by='number_of_transactions',ascending=False) 
        
        if recommendations_df.shape[0] == 0:
            return []

        recommendations = recommendations_df['product'].tolist()[:top_k]  
        
        return recommendations
    
    def get_apriori_recommendations(self, products, top_k = 5):
        recommendation_list = []
        
        for product in products:
            if product in self.apriori_recommendations:
                recommendation_list += self.apriori_recommendations[product]
                
        # Sort recommendation list by confidence
        recommendation_list = sorted(recommendation_list, key = lambda x:x['confidence'], reverse=True)
        
        recommendations = []
        recommendation_per_category = {}
        for recommendation in recommendation_list:
            if recommendation in recommendations:
                continue
            
            # Limit 2 recommendations per categories
            product_category = recommendation['product_categories']
            if product_category not in recommendation_per_category:
                recommendation_per_category[product_category] = 0
                
            if recommendation_per_category[product_category] > 2:
                continue
            
            recommendation_per_category[product_category] += 1
            
            # Add recommendations
            recommendations.append(recommendation['product'])
            
            if len (recommendations) >= top_k:
                break
            
        return recommendations
    
    def recommendation_classification (self,messages ):  
        memssages = deepcopy(messages)
        system_prompt = """ You are a helpful AI assistant for a coffee shop application which serves drinks and pastries. We have 3 types of recommendations:

        1. Apriori Recommendations: These are recommendations based on the user's order history. We recommend items that are frequently bought together with the items in the user's order.
        2. Popular Recommendations: These are recommendations based on the popularity of items in the coffee shop. We recommend items that are popular among customers.  
        3. Popular Recommendations by Category: Here the user asks to recommend them product in a category. Like what coffee do you recommend me to get?. We recommend items that are popular in the category of the user's requested category.
        
        Here is the list of items in the coffee shop:
        """+ ",".join(self.products) + """
        Here is the list of Categories we have in the coffee shop:
        """ + ",".join(self.product_categories) + """

        Your task is to determine which type of recommendation to provide based on the user's message.

        Your output should be in a structured json format like so. Each key is a string and each value is a string. Make sure to follow the format exactly:
        {
        "chain of thought": Write down your critical thinking about what type of recommendation is this input relevant to.
        "recommendation_type": "apriori" or "popular" or "popular by category". Pick one of those and only write the word.
        "parameters": This is a  python list. It's either a list of of items for apriori recommendations or a list of categories for popular by category recommendations. Leave it empty for popular recommendations. Make sure to use the exact strings from the list of items and categories above.
        }
        """
        
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]
        
        chatbot_response = get_llm_response(self.client, self.model_name, input_messages) 
        chatbot_response = double_check_json_output(self.client, self.model_name, chatbot_response)  
        
        output= self.postprocess_classification(chatbot_response)
        
        return output
    

    def postprocess_classification(self, output: str) -> Dict[str, Any]:
        # FIX: This method is now more robust. It finds the JSON within the string
        # and handles parsing errors to prevent crashes.
        try:
            # Find the start and end of the JSON object
            start_index = output.find('{')
            end_index = output.rfind('}') + 1
            
            if start_index == -1 or end_index == 0:
                print("Error: No JSON object found in the output.")
                return {'recommendation_type': None, 'parameters': []}

            json_str = output[start_index:end_index]
            output_dict = json.loads(json_str)
            
            # To handle both "chain_of_thought" and "chain of thought" from the LLM
            processed_dict = {
                'recommendation_type': output_dict.get('recommendation_type'),
                'parameters' : output_dict.get('parameters', [])
            }
            return processed_dict

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Problematic output string: {output}")
            # Return a default dictionary to avoid crashing downstream
            return {'recommendation_type': None, 'parameters': []}
    
    def get_recommendations_from_order(self,messages,order):
        memssages = deepcopy(messages)
        products = []
        for product in order:
            products.append(product['item'])
        
        recommendations = self.get_apriori_recommendations(products)
        recommendationn_str = ', '.join(recommendations)
        
        system_prompt = f"""
        You are a helpful AI assistant for a coffee shop application which serves drinks and pastries.
        Your task is to recommend items to the user based on their input message. And respond in a friendly but concise way. And put it an unordered list with a very small description.

        I will provide what items you should recommend to the user based on their order in the user message. 
        """
        
        prompt = f"""
        {messages[-1]['content']}
        
        Please recommend me those items exactly: {recommendationn_str}
        """
        
        messages[-1]['content'] = prompt 
        input_messages= [{'role':'system', 'content':system_prompt}] + messages[-3:]
        
        chatbot_response = get_llm_response(self.client, self.model_name, input_messages)
        output = self.postprocess(chatbot_response)
        
        return output
    
    def get_response(self,messages):
        messages = deepcopy(messages)
    
        recommendation_classification = self.recommendation_classification(messages)
        recommendation_type = recommendation_classification['recommendation_type']

        recommendations = []

        if recommendation_type == 'apriori':
            recommendations = self.get_apriori_recommendations(recommendation_classification['parameters'])
        elif recommendation_type == 'popular':
            recommendations = self.get_popular_recommendation()
        elif recommendation_type == 'popular by category':
            recommendations = self.get_popular_recommendation( recommendation_classification['parameters'])
            
        if recommendations == []:
            return {'role':"assistant", "content":"sorry I can't help with that recommendation. Can I help you with something else"}
        
        
        # Responsne to use
        recommendations_str = ", ".join(recommendations)
        
        system_prompt = f"""
        You are a friendly and helpful AI assistant for a coffee shop called Sama Sama Coffee.
        The user has just asked for a recommendation. Your task is to answer their last message.
        Use the following list of items to form your recommendation response. Present them in a friendly, conversational way .

        Items to recommend: {recommendations_str}
        """ 
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]
        
        print("input_messages: ", input_messages)
        chatbot_response = get_llm_response(self.client, self.model_name, input_messages)
        print("OUTPUT:", chatbot_response )
        output = self.postprocess(chatbot_response)
        
        return output
        
        
    
    def postprocess(self,output):
        output = {
            'role':'assistant',
            'content': output,
            'memory' : {'agent':'recommendation_agent'}
        }
        
        return output