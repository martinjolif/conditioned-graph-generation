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
    encoded = tokenizer(graph_text_description,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt')

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**encoded)
        # Use the [CLS] token embedding (first token)
        sentence_embedding = outputs.last_hidden_state[:, 0, :]

    return sentence_embedding.squeeze(1)


def bert_encoding_from_graph_filename(file):
    stats = []
    fread = open(file, "r")
    line = fread.read()
    line = line.strip()
    stats = bert_encoding_from_graph_text_description(line)
    fread.close()
    return stats