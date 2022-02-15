import torch
from fastapi import FastAPI
from transformers import DistilBertTokenizer, BertForSequenceClassification

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = BertForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels = 2)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model.load_state_dict(torch.load("multilingual_hate_speech_model.pth", map_location=torch.device('cpu')))
model.to(device)
model.eval()

app = FastAPI()

def prediction(text):
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=64,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors = "pt",
    )
    ids = encoded['input_ids']
    masks = encoded['attention_mask']
    ids = ids.to(device, dtype=torch.long)
    masks = masks.to(device, dtype=torch.long)
    with torch.no_grad():
        output = model(
            input_ids=ids,
            attention_mask=masks,
            token_type_ids=None,
        )
        return torch.sigmoid(output[0]).detach().cpu().numpy()

@app.get("/ping")
def ping():
    return {"message": "pong!"}


@app.get("/predict/{sentence}")
def predict(sentence: str):
    pred = prediction(sentence)[0][0]
    hate_or_not = pred
    if pred < 0.5:
        pred = "hate"
    elif pred == 0.5:
        pred = "neutral"
    else:
        pred = "not hate"
    return {"accuarcy": str(hate_or_not), "message": str(pred)}