import torch
import open_clip
import os

model_name='ViT-B-32'
pretrained='openai'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
model, _, preprocess =  open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
tokenizer = open_clip.get_tokenizer(model_name)

model.to(device)
model.eval()

def extract_encoding_from_graph_text_description(graph_text_description):
        "return the embedding of a graph description with a shape of (1,512)"
        text = tokenizer([graph_text_description], context_length=77).to(device)
        x = model.encode_text(text)

        return x.detach().cpu().numpy()

def extract_encoding_from_graph_filename(file):
        stats = []
        fread = open(file, "r")
        line = fread.read()
        line = line.strip()
        stats = extract_encoding_from_graph_text_description(line)
        fread.close()
        return stats