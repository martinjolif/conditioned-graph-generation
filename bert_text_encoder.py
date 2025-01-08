from transformers import BertTokenizer, BertModel
import torch

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()

def bert_encoding_from_graph_text_description(graph_text_description):
    "return the embedding of a graph description with a shape of "
    encoding = tokenizer.batch_encode_plus([graph_text_description],  # List of input texts
                                           padding=True,  # Pad to the maximum sequence length
                                           truncation=True,  # Truncate to the maximum sequence length if necessary
                                           return_tensors='pt',  # Return PyTorch tensors
                                           add_special_tokens=True  # Add special tokens CLS and SEP
                                           )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        #sentence_embeddings = outputs.last_hidden_state.mean(dim=1)
        # 0 refers to the first token [CLS] which is a special token used to represent the entire sentence
        sentence_embeddings = outputs.last_hidden_state[:,0,:]

    return sentence_embeddings


def bert_encoding_from_graph_filename(file):
    stats = []
    fread = open(file, "r")
    line = fread.read()
    line = line.strip()
    stats = bert_encoding_from_graph_text_description(line)
    fread.close()
    return stats