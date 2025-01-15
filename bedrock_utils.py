import boto3
import json

def invoke_bedrock_model(model_name, messages, model_params=None, max_tokens=None):
    """
    Invoke a Bedrock model with the given messages using SageMaker's execution role.

    Args:
    model_name (str): The name of the Bedrock model to use.
    messages (list): A list of message dictionaries, each containing 'role' and 'content'.
    model_params (dict, optional): Additional model parameters.
    max_tokens (int, optional): Maximum number of tokens in the response.

    Returns:
    str: The model's response.
    """
    # Initialize the Bedrock client
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-west-2'
    )

    # Prepare the request body
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "max_tokens": max_tokens if max_tokens else 2048,
        "temperature": 0.7,
        "top_p": 1,
    }

    # Add any additional model parameters
    if model_params:
        body.update(model_params)

    # Convert the body to a JSON string
    body = json.dumps(body)

    invoke_args = {
        "modelId": model_name,
        "body": body,
        "contentType": "application/json",
        "accept": "application/json"
    }

    # Make the API call
    try:
        response = bedrock.invoke_model(**invoke_args)
        # Parse and return the response
        response_body = json.loads(response.get("body").read())
        return response_body
    except Exception as e:
        print(f"Error invoking Bedrock model: {str(e)}")
        return None

def main():
    model_name = "anthropic.claude-3-sonnet-20240229-v1:0"  # Updated model name
    messages = [
        {"role": "user", "content": "Explain the concept of machine learning in simple terms."}
    ]

    response = invoke_bedrock_model(model_name, messages)
    if response:
        print("Model response:", response)
    else:
        print("Failed to get a response from the model.")

if __name__ == "__main__":
    main()