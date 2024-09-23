import os

import pandas as pd
from LLMS import \
    call_model  # Import the model calling function from the LLMS file

# CSV file path
csv_file_path = r"E:\github\EUQA\test2.csv"   # Replace with the path to your CSV file
folder_path = r"E:\github\EUQA"  # Replace with the folder path that contains the long documents

# Read the test CSV file
test_df = pd.read_csv(csv_file_path)

# Accuracy calculation
total_tests = 0
correct_predictions = 0

# Helper function: Extract text content from a .txt file
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Helper function: Extract the predicted answer from the model's response
def extract_answer_from_response(response):
    if hasattr(response, 'text'):
        response_text = response.text
    elif hasattr(response, 'choices'):
        response_text = response.choices[0].message['content']
    else:
        response_text = str(response)

    # Assuming the model's output contains a clear answer, extract the answer (adjust parsing logic if needed)
    return response_text.strip()

# Loop through each row and calculate the model's accuracy
for index, row in test_df.iterrows():
    article_id = row['Article_ID']  # Corresponding article number
    question = row['Question']
    correct_answer = row['Correct_Answer']  # Used to compare with the model's predicted answer

    # Find the corresponding long document based on the article ID
    document_file = os.path.join(folder_path, f"{article_id}.txt")

    # Read the long document content
    long_document = extract_text_from_txt(document_file)

    # Construct the prompt
    prompt = (
        f"You are an A assistant. Your task is to read the long text provided and answer questions based on the plot "
        f"and content, and provide a reason. The questions include multiple-choice questions and open-ended questions. "
        f"For the choice questions, I will tell you the number of correct options. For the open-ended questions, your answer "
        f"should be no more than 100 words. Please do NOT output any extra content. Sample Input (format only):\n\n"
        f"Document {article_id}: {long_document}.\n"
        f"Please read the entire document and provide an answer to the following question: \"{question}\". "
        f"Select the correct answer based on the information provided in the document."
    )

    # Call the model
    model_response = call_model("glm-4-0520", prompt)

    # Extract the predicted answer from the model's response
    predicted_answer = extract_answer_from_response(model_response)

    # Compare the model's predicted answer with the correct answer
    if predicted_answer.lower() == correct_answer.lower():
        correct_predictions += 1
    total_tests += 1

# Calculate and output accuracy
accuracy = correct_predictions / total_tests if total_tests > 0 else 0
print(f"Model accuracy: {accuracy:.2f}")
