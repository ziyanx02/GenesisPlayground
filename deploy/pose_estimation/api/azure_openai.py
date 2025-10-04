import os
import base64
from openai import AzureOpenAI
from mimetypes import guess_type

client = AzureOpenAI(
    # api_key="7EPgiBlH9dLjPZzVip6phrwmP0X67BOOAAZeh9kbEHDL1CPPUdLgJQQJ99BCACYeBjFXJ3w3AAABACOGAqyR",  
    # api_key="10wYQ8PEOZeRm7Z1DyIecJygKxh3uQUDBDdPdWovEKnGptU3eulAJQQJ99BDAC4f1cMXJ3w3AAABACOGdFuw", 
    api_key="7cyULKybey1ZltRzKMTiqzMAIUe5L0VzHTc1VMYWtLDP1z2EKHOZJQQJ99BDACHYHv6XJ3w3AAABACOGCWcD",
    api_version="2024-12-01-preview",
    # azure_endpoint="https://ziyanx-openai.openai.azure.com/",
    # azure_endpoint="https://3spatialo1.openai.azure.com/",
    azure_endpoint="https://spatial-1.openai.azure.com/",
)

# deployment_name='gpt-4o' #This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment. 
deployment_name='o1' #This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment. 

def complete(messages):
    print("Completing...")
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
    )
    print("Completed.")
    return response.choices[0].message.content

def local_image_to_data_url(image_path):
    """
    Get the url of a local image
    """
    mime_type, _ = guess_type(image_path)

    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{base64_encoded_data}"

if __name__ == "__main__":
    # Send a completion call to generate an answer
    print('Sending a test completion job')
    messages=[
        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
        {"role": "user", "content": "Who were the founders of Microsoft?"}
    ]
    print(complete(messages))