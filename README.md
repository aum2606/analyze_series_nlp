# Multimedia Analysis & Interaction Framework

This project is a comprehensive multimedia analysis tool built with Gradio that provides functionality for theme classification, character network visualization, text classification, and character chatbot interaction. It leverages advanced machine learning models for zero-shot classification, named entity recognition (NER), and text generation.

## Features

1. **Theme Classification**  
   - Classifies themes in subtitles or scripts using zero-shot classification models.
   - Outputs a bar chart displaying theme scores for insights into series or script themes.

2. **Character Network Visualization**  
   - Extracts named entities from subtitles or scripts and generates a relationship network of characters.
   - Visualizes the character network as an interactive HTML graph.

3. **Text Classification with LLMs**  
   - Classifies textual data using a pre-trained large language model.
   - Designed for custom tasks such as Jutsu classification in specific datasets.

4. **Character Chatbot**  
   - A conversational chatbot based on a fine-tuned LLM (e.g., Naruto_Llama-3-8B).
   - Supports dynamic conversations with historical context.

## Prerequisites

Ensure the following are installed:

- Python 3.8 or higher
- Gradio
- Pydantic
- Additional dependencies listed in `requirements.txt`

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/aum2606/multimedia-analysis.git
   cd multimedia-analysis
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Configure your environment:
    Add your Hugging Face API token to a .env file as follows:
    ```bash
    huggingface_token=your_huggingface_api_token
    ```

# Usage
Run the application with the following command:
```bash
python gradio_app.py
```
This will launch a Gradio interface where you can interact with the tools.

# File structure
- **gradio_app.py**: Main script for launching the Gradio interface.
- **character_network/**: Contains modules for named entity recognition and character network generation.

    - **character_network_generator.py**: Generates character relationships and visualizes them.
    - **named_entity_recognizer.py**: Extracts named entities from scripts.

- **theme_classifier/**: Module for classifying themes in subtitles or scripts.

    - **theme_classifier.py**: Implements zero-shot theme classification.

- **text_classification.py**: Script for text classification using LLMs.
- **character_chatbot.py**: Handles character chatbot interactions using an LLM.


# Functional Details
1. **Theme Classification**

    Input: List of themes, subtitles path, and save path.
    Output: A bar chart of theme scores.

2. **Character Network**

    Input: Subtitles path and NER save path.
    Output: An interactive HTML graph of character relationships.

3. **Text Classification**

    Input: Model path, data path, and text to classify.
    Output: Classified text category.

4. **Character Chatbot**

    Input: User message and conversation history.
    Output: Chatbot response based on the selected character model.


# Example Usage

1.  **Theme Classification**:
        Input: Themes: "love, conflict, action", Path: "data/subtitles.srt".
        Output: Bar chart of scores for "love", "conflict", and "action".

2.  **Character Network**:
        Input: Subtitles path: data/subtitles.srt, NER path: data/ner.json.
        Output: HTML visualization of character connections.

3.  **Text Classification**:
        Input: Custom text and model for Jutsu classification.
        Output: Classified Jutsu type.

4.  **Character Chatbot**:
        Input: Chat message like "What are your thoughts on Naruto?".
        Output: Chatbot-generated response.


# Future Enhancements
- Expand theme classification to include sentiment analysis.
- Enable live-stream input for character networks.
- Add multi-language support for subtitles and scripts.

# License

This project is licensed under the MIT License. See the LICENSE file for more details.

# Acknowledgments

- Gradio for creating an intuitive user interface.
- Hugging Face for providing pre-trained language models.
- Libraries like Pydantic and OpenAI for simplifying development.