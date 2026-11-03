
import gradio as gr

from src.predict import load_model, predict_text


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("Initializing BERT News Classifier...")

tokenizer, model = load_model()

print("Application ready!")


# ============================================================
# 2. PREDICTION FUNCTION
# ============================================================

def classify_news(text):

    try:
        result = predict_text(
            text=text,
            tokenizer=tokenizer,
            model=model
        )

        prediction = (
            f"Predicted Category: {result['predicted_category']}\n"
            f"Confidence: {result['confidence']:.2f}%"
        )

        # Gradio Label expects category scores as decimals
        probabilities = {
            category: probability / 100
            for category, probability
            in result["probabilities"].items()
        }

        return prediction, probabilities

    except ValueError as error:
        return str(error), {}

    except Exception as error:
        print(f"Prediction error: {error}")
        return f"An error occurred: {str(error)}", {}


# ============================================================
# 3. EXAMPLE ARTICLES
# ============================================================

examples = [
    [
        "The United States government announced new international "
        "trade policies today."
    ],
    [
        "India defeated Australia in an exciting cricket match."
    ],
    [
        "The stock market increased after major technology companies "
        "reported strong quarterly earnings."
    ],
    [
        "Scientists developed a new artificial intelligence technology "
        "that could transform medical research."
    ]
]


# ============================================================
# 4. GRADIO INTERFACE
# ============================================================

custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

#main-title {
    text-align: center;
}

#classify-button {
    font-weight: bold;
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:

    gr.Markdown(
        """
        # 📰 BERT News Classifier

        ### Intelligent News Classification using Fine-Tuned BERT

        Classify news articles into four categories using a
        fine-tuned BERT model.

        **Categories:** 🌍 World | 🏀 Sports | 💼 Business | 🔬 Sci/Tech
        """,
        elem_id="main-title"
    )

    gr.Markdown(
        """
        ---
        ### Enter Your News Article
        """
    )

    news_input = gr.Textbox(
        label="News Article",
        placeholder="Paste your news article here...",
        lines=7
    )

    classify_button = gr.Button(
        "🚀 Classify News",
        variant="primary",
        elem_id="classify-button"
    )

    gr.Markdown("### 🎯 Prediction Result")

    prediction_output = gr.Textbox(
        label="Predicted Category",
        interactive=False,
        lines=2
    )

    gr.Markdown("### 📊 Category Probabilities")

    probability_output = gr.Label(
        label="Classification Confidence",
        num_top_classes=4
    )

    gr.Markdown("### 🧪 Try Example Articles")

    gr.Examples(
        examples=examples,
        inputs=news_input
    )

    classify_button.click(
        fn=classify_news,
        inputs=news_input,
        outputs=[
            prediction_output,
            probability_output
        ]
    )

    news_input.submit(
        fn=classify_news,
        inputs=news_input,
        outputs=[
            prediction_output,
            probability_output
        ]
    )

    gr.Markdown(
        """
        ---

        **Model:** BERT Base Uncased

        **Dataset:** AG News

        **Test Accuracy:** 94.38%

        **Framework:** PyTorch | Hugging Face Transformers | Gradio
        """
    )


# ============================================================
# 5. LAUNCH APPLICATION
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860
    )
