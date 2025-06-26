import openai
from openai import AzureOpenAI
import json

def api_key() :
    file = open('api_key.txt', 'r')
    api_key = file.read()
    return api_key

def api_version():
    file = open('api_version.txt', 'r')
    api_version = file.read()
    return api_version

def azure_endpoint():
    file = open('azure_endpoint.txt', 'r')
    azure_endpoint = file.read()
    return azure_endpoint

def model():
    file = open('model.txt', 'r')
    model = file.read()
    return model

def collection_of_metadata_tags() :

    metadata_tags = """

    Company : Amazon, Goldmann Sachs, Bajaj, Microsoft, Google, LEK_Consulting, Nvidia, Sprinklr, Texas_Instruments ;

    Branch : Computer Science and Engineering (CSE), Mathematics and Computing (MnC), Electrical Engineering (EE), Electronics and Electrical Communications Engineering (EECE), Chemical Engineering (Chem.), Civil Engineering (CE), Mechanical Engineering (ME), Metallurgical and Materials Engineering (Metallurgy), Exploration Geophysics (EX), Instrumentation Engineering , Industrial and Systems Engineering, Chemistry, Physics, Aerospace Engineering (AE), Agriculture and Food Engineering (Agri.), Applied Geology, Architecture (Architect), Artificial Intelligence (AI), Biotech and Biochemical Engineering (Biotech.), Economics, Manufacturing Science and Engineering, Mining Engineering, Ocean Engineering and Naval Architecture (OENA) ;

    Department : Computer Science and Engineering, Mathematics, Chemistry, Chemical Engineering, Electrical Engineering, Biotechnology Department, Aerospace Engineering, Electronics and Electrical Communications Engineering, Physics, Mining, Metallurgical, Civil Engineering, Industrial, Instrumentation ;

    Gender : Male, Female ;

    """
    return metadata_tags

def input_query():
    query = input()
    return query

query_string = input_query()

def query(query_string=None):
    if query_string is None:
        query_string = input_query()

    return query_string

def prompt_to_llm() :

    prompt = f"""You are given a string which contains some metadata tags and their values. A semicolon depicts the end of a particular tag. The string which contrains metadata tags is : {collection_of_metadata_tags()}.\n\n Now, you are given a query which is: {query(query_string)}\n\n. Your task is to identify the metadata tags which are present in the query string and their values as well. You must not provide more than 5 metadata tags. Your output must be a json format response in which metadata tags are mapped with their values as given in the query, you MUST NOT provide anything else as output. You must not write word like python, json in the output.

    """
    return prompt

def response_by_llm() :

    client = AzureOpenAI(
        api_key = api_key(),
        api_version = api_version(),
        azure_endpoint = azure_endpoint()
    )

    response = client.chat.completions.create(
        model=model(), # model = "deployment_name".
        messages=[
            {"role": "system", "content": f"{prompt_to_llm()}"},
            {"role": "user", "content": query(query_string)}
        ]
    )

    return response

extracted_tags = response_by_llm().choices[0].message.content
tags = json.loads(extracted_tags)

def main() :
    return tags
