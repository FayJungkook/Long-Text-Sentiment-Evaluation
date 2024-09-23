import os

import pandas as pd
from LLMS import \
    call_model  # Import the model calling function from the LLMS file

# Path to the CSV file
test_file_path = r"E:\github\SC\English TV series\test1.csv"   # Replace with the path to your CSV file
folder_path = r"E:\github\SC\English TV series"  # Replace with the path to the folder containing the long documents

# Read the test CSV file
test_df = pd.read_csv(test_file_path)

# Emotion label range
emotion_labels = ["neutral", "fear", "joy", "sadness", "anger"]

# Accuracy calculation
total_tests = 0
correct_predictions = 0

# Helper function: Extract text from a .txt document
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Helper function: Parse the emotion label from the model response
def extract_label_from_response(response):
    if hasattr(response, 'text'):
        response_text = response.text
    elif hasattr(response, 'choices'):
        response_text = response.choices[0].message['content']
    else:
        response_text = str(response)

    reason = "The model's reasoning is unknown."
    label = "neutral"
    for label_option in emotion_labels:
        if label_option in response_text.lower():
            label = label_option
            break
    return reason, label

# Iterate through each row in the table to calculate accuracy
for index, row in test_df.iterrows():
    season = row['Season']
    episode = row['Episode']
    test_sentence = row['Test_Sentence']
    gold_label = row['Gold_Label']  # Gold label for comparison with model prediction

    # Find the corresponding long document based on Season and Episode
    document_file = os.path.join(folder_path, f"S{season}E{episode}.txt")

    # Read the content of the long document
    long_document = extract_text_from_txt(document_file)

    # Construct the updated prompt in English
    prompt = (f"Document S{season}E{episode}: {long_document}.\n"
              f"You are an AI assistant. Your task is to read the long text provided and select one of the following emotions as a sentiment label. "
              f"Please select a sentiment label for the following sentence, output the sentence and label, and provide a justification based on the context. "
              f"Please do NOT output any extra content. Sample Input (format only): \"{test_sentence}\".\n"
              f"The emotion label range is {', '.join(emotion_labels)}.")

    # Call the model
    model_response = call_model("glm-4-0520", prompt)

    # Get the model's reasoning and emotion label
    reason, predicted_label = extract_label_from_response(model_response)

    # Compare the model's predicted label with the gold label
    if predicted_label == gold_label.lower():
        correct_predictions += 1
    total_tests += 1

# Calculate and output accuracy
accuracy = correct_predictions / total_tests if total_tests > 0 else 0
print(f"Model accuracy: {accuracy:.2f}")
