import torch
import numpy as np
import open_clip
import os
#from transformers import CLIPModel, AutoTokenizer, CLIPProcessor

model_name='ViT-B-32'
pretrained='openai'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
model, _, preprocess =  open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
tokenizer = open_clip.get_tokenizer(model_name)

model.to(device)
model.eval()

#give your instruction here

def encode_graph_text_description(graph_text_description):
        "return the encoding of a graph description with a shape of (1,512)"
        text = tokenizer([graph_text_description], context_length=77).to(device)
        x = model.encode_text(text)

        return x.detach().cpu().numpy()