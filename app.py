from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load model
model = load_model("multilingual_lstm_model.h5")
print("Model input shapes:", model.input_shape)

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    data = pickle.load(f)

# If you saved (tokenizer, max_len)
if isinstance(data, tuple):
    tokenizer = data[0]
    max_len = data[1]
else:
    tokenizer = data

max_length = 11 # Change according to training

# Language tokens (used during training)
language_tokens = {
    "english": "<en>",
    "hindi": "<hi>",
    "punjabi": "<pa>"
}

def translate_text(input_text, source_lang, target_lang):

    # Add language tag
    tagged_input = f"<to_{target_lang}> " + input_text

    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([tagged_input])
    encoder_input = pad_sequences(sequence, maxlen=10, padding='post')

    # Get start token index
    start_token = tokenizer.word_index["<start>"]
    end_token = tokenizer.word_index["<end>"]

    # Prepare decoder input
    decoder_input = np.zeros((1, 11))
    decoder_input[0, 0] = start_token

    output_sentence = []

    for i in range(1, 11):

        prediction = model.predict([encoder_input, decoder_input], verbose=0)

        predicted_id = np.argmax(prediction[0, i-1])

        if predicted_id == end_token or predicted_id == 0:
            break

        word = tokenizer.index_word.get(predicted_id, "")

        output_sentence.append(word)

        decoder_input[0, i] = predicted_id

    return " ".join(output_sentence)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/translate", response_class=HTMLResponse)
async def translate(
    request: Request,
    source_lang: str = Form(...),
    target_lang: str = Form(...),
    text: str = Form(...)
):
    result = translate_text(text, source_lang, target_lang)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result
        }
    )