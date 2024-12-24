import os
import gradio as gr
from character_netwrok.character_network_generator import CharacterNetworkGenerator
from character_netwrok.named_entity_recognizer import NamedEntityRecognizer
from text_classification import JutsuClassifier
from theme_classifier.theme_classifier import ThemeClassifier
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()


class MyModel(BaseModel):
    # Your fields here
    
    class Config:
        arbitrary_types_allowed = True


def get_themes(theme_list_str,subtitles_path,save_path):
    theme_list = theme_list_str.split(',')
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path,save_path)

    # Remove dialogue from the theme list
    theme_list = [theme for theme in theme_list if theme != 'dialogue']
    output_df = output_df[theme_list]

    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ['Theme','Score']

    output_chart = gr.BarPlot(
        output_df,
        x="Theme",
        y="Score",
        title="Series Themes",
        tooltip=["Theme","Score"],
        vertical=False,
        width=500,
        height=260
    )

    return output_chart


def get_character_network(subtitles_path,ner_path):
    ner = NamedEntityRecognizer()
    ner_df = ner.get_ners(subtitles_path,ner_path)

    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)
    return html


def classify_text(text_classifcation_model,text_classifcation_data_path,text_to_classify):
    jutsu_classifier = JutsuClassifier(model_path = text_classifcation_model,
                                       data_path = text_classifcation_data_path,
                                       huggingface_token = os.getenv('huggingface_token'))
    
    output = jutsu_classifier.classify_jutsu(text_to_classify)
    output = output[0]
    
    return output



def main():
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme classification (Zero shot classifiers)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot()
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes")
                        subtitles_path = gr.Textbox(label="subtitles or script path")
                        save_path = gr.Textbox(label="Save path")
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(get_themes,inputs=[theme_list,subtitles_path,save_path],outputs=[plot])
                        
        # Character Network Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network (NERs and Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        subtitles_path = gr.Textbox(label="Subtutles or Script Path")
                        ner_path = gr.Textbox(label="NERs save path")
                        get_network_graph_button = gr.Button("Get Character Network")
                        get_network_graph_button.click(get_character_network, inputs=[subtitles_path,ner_path], outputs=[network_html])

       # Text Classification with LLMs
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Text Classification with LLMs</h1>")
                with gr.Row():
                    with gr.Column():
                        text_classification_output = gr.Textbox(label="Text Classification Output")
                    with gr.Column():
                        text_classifcation_model = gr.Textbox(label='Model Path')
                        text_classifcation_data_path = gr.Textbox(label='Data Path')
                        text_to_classify = gr.Textbox(label='Text input')
                        classify_text_button = gr.Button("Clasify Text (Jutsu)")
                        classify_text_button.click(classify_text, inputs=[text_classifcation_model,text_classifcation_data_path,text_to_classify], outputs=[text_classification_output])
                        
                    
    iface.launch(share=True)
                    
                    
    
if __name__ == "__main__":
    main()