from turtle import title
import gradio as gr
from transformers import pipeline
import numpy as np
from PIL import Image




pipe = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
images="caesar.jpg"

def shot(image, labels_text):
    PIL_image = Image.fromarray(np.uint8(image)).convert('RGB')
    labels = labels_text.split(",")
    res = pipe(images=PIL_image, 
           candidate_labels=labels,
           hypothesis_template= "This is a photo of a {}")
    return {dic["label"]: dic["score"] for dic in res}
    
# iface = gr.Interface(shot, 
#                     ["image", "text"], 
#                     "label", 
#                     examples=[["caesar.jpg", "dog,cat,bird"], 
#                               ["benny.jpg", "dog,cat,bird"]])

demo = gr.Interface(
     shot, 
     ["image", "text"], 
"label",
    examples=[["caesar.jpg", "dog,cat,bird"], 
                ["benny.jpg", "dog,cat,bird"]]
)

# with gr.Blocks() as demo:
#     with gr.Row(equal_height=True):
#         with gr.Column():
#             input = gr.Image()
#         with gr.Column():
#             examples=[["caesar.jpg", "dog,cat,bird"], 
#                         ["benny.jpg", "dog,cat,bird"]]
#             output = gr.Textbox()



if __name__ == "__main__":
    demo.launch()



# if __name__ == "__main__":
#     iface.launch()