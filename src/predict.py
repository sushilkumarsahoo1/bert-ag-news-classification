
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "sushil9876/bert-ag-news-classifier"

MAX_LENGTH = 256

ID_TO_LABEL = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}


# --------------------------------------------------
# Device Configuration
# --------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()


# --------------------------------------------------
# Load Fine-Tuned Model
# --------------------------------------------------

def load_model(model_name=MODEL_NAME):

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name
    )

    model.to(DEVICE)
    model.eval()

    print(f"Model loaded successfully on {DEVICE}")

    return tokenizer, model


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_text(text, tokenizer, model):

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please provide a valid news article.")

    # Tokenization
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    # Move tensors to device
    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # Model inference
    with torch.inference_mode():

        outputs = model(**inputs)

        probabilities = F.softmax(
            outputs.logits,
            dim=-1
        )[0]

    # Get predicted class
    predicted_id = torch.argmax(probabilities).item()

    predicted_category = model.config.id2label.get(
        predicted_id,
        ID_TO_LABEL[predicted_id]
    )

    confidence = probabilities[predicted_id].item() * 100

    # All category probabilities
    category_probabilities = {
        model.config.id2label.get(i, ID_TO_LABEL[i]):
        round(probabilities[i].item() * 100, 2)
        for i in range(len(probabilities))
    }

    return {
        "predicted_category": predicted_category,
        "confidence": round(confidence, 2),
        "probabilities": category_probabilities
    }


# --------------------------------------------------
# Batch Prediction Function
# --------------------------------------------------

def predict_batch(texts, tokenizer, model):

    if not texts:
        raise ValueError("Please provide at least one article.")

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every article must contain valid text.")

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model(**inputs)

        probabilities = F.softmax(
            outputs.logits,
            dim=-1
        )

    results = []

    for probs in probabilities:

        predicted_id = torch.argmax(probs).item()

        predicted_category = model.config.id2label.get(
            predicted_id,
            ID_TO_LABEL[predicted_id]
        )

        category_probabilities = {
            model.config.id2label.get(i, ID_TO_LABEL[i]):
            round(probs[i].item() * 100, 2)
            for i in range(len(probs))
        }

        results.append({
            "predicted_category": predicted_category,
            "confidence": round(
                probs[predicted_id].item() * 100, 2
            ),
            "probabilities": category_probabilities
        })

    return results


# --------------------------------------------------
# Test the Prediction Pipeline
# --------------------------------------------------

if __name__ == "__main__":

    tokenizer, model = load_model()

    sample_news = """
    The Los Angeles Lakers defeated the Boston Celtics
    in an exciting basketball match last night.
    """

    result = predict_text(
        sample_news,
        tokenizer,
        model
    )

    print("\nPrediction Results")
    print("------------------")

    print("Category:", result["predicted_category"])
    print("Confidence:", result["confidence"], "%")

    print("\nCategory Probabilities:")

    for category, probability in result["probabilities"].items():
        print(f"{category}: {probability}%")
