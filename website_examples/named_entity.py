# Stolen from: https://huggingface.co/spaces/AyushDey/Named_Entity_Recognition/tree/main

import gradio as gr
from transformers import pipeline

ner = pipeline('ner')

def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity'].startswith('I-') and merged_tokens[-1]['entity'].endswith(token['entity'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens

examples = [
    "Johann Carl Friedrich Gauss was a German mathematician, geodesist, and physicist who made significant contributions to many fields in mathematics and science.",
    'At Los Alamos, which was isolated for security, Feynman amused himself by investigating the combination locks on the cabinets and desks of physicists. He often found that they left the lock combinations on the factory settings, wrote the combinations down, or used easily guessable combinations like dates.'
]

def named(input):
    output = ner(input)
    merged_word = merge_tokens(output)
    return {'text': input, 'entities': merged_word}

init = 'At Los Alamos, which was isolated for security, Feynman amused himself by investigating the combination locks on the cabinets and desks of physicists. He often found that they left the lock combinations on the factory settings, wrote the combinations down, or used easily guessable combinations like dates.'

demo = gr.Interface(fn=named, 
                 inputs=[gr.Textbox(label="Text input", value=init, lines= 2)],
                 outputs=[gr.HighlightedText(label='Labelled text')])


#Launching the gradio app
if __name__ == "__main__":
  demo.launch(debug=True)