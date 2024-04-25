# User Guide

This Python script demonstrates the use of the `TextGeneration` and `get_embeddings` functions for text generation and obtaining embeddings from training data.

## Prerequisites

Before running the script, make sure you have Python installed on your system. You can download Python from the official website: [https://www.python.org/downloads/](https://www.python.org/downloads/).

## Installing Dependencies

To run the script, you'll need to install the required dependencies. You can do this by running the following command in your terminal:

``` pip install -r requirements.txt ```


## Using the Script

To use the text generation and embeddings functions, follow the steps below:

1. Ensure you have a file containing your training data in JSON format. The file should be in the following format:

   ```json
   {
       "sentence_1": "target_1",
       "sentence_2": "target_2",
       ...
   }
   
where each key is a training sentence and each value is the corresponding target



Place your training data file in the same directory as the main.py script.

Run the main.py script using the following command:
```python main.py```

This will execute the TextGeneration and get_embeddings functions with the specified training data file.
