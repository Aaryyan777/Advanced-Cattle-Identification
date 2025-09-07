
# Advanced AI Classification Pipeline

This project implements a robust, two-stage AI pipeline for accurate image classification, featuring a state-of-the-art, zero-shot filter to handle real-world, out-of-distribution images gracefully.

---

## 1. The Challenge: Handling Real-World Images

Initially, our expert model for classifying cattle breeds (`EfficientNetV2-S`) was highly accurate on its specific task. However, it exhibited a common weakness of specialized models: it performed poorly when presented with images outside its training distribution. For example, when given a photo of a person or a car, it would still try to classify it as a cattle breed, leading to nonsensical results.

This highlighted the need for a smart "gatekeeper" or filter that could identify and reject irrelevant images *before* they reached our expert model.

## 2. Our Solution: A Two-Stage Pipeline with a Zero-Shot Filter

Instead of training a separate, custom filter model (which proved to be a complex and brittle process), we pivoted to a more modern and powerful solution: a **two-stage pipeline** leveraging a pre-trained, zero-shot classification model.

### Stage 1: The CLIP Filter (The "Gatekeeper")

This is the core of the new, robust system. We use **OpenAI's CLIP model** (`clip-vit-large-patch14`), a foundational model that has a deep, nuanced understanding of the relationship between images and text.

#### How it Works: Zero-Shot Classification

Instead of being trained on a fixed set of categories, CLIP can classify an image against arbitrary text descriptions *without any training*. This is called **zero-shot classification**.

1.  When an image is uploaded, we don't just ask the model "is this a cow?".
2.  Instead, we ask it to choose between two descriptive labels:
    *   `"a photo of a cow, bull, or cattle"`
    *   `"a photo of something else (person, object, landscape, other animal)"`
3.  CLIP analyzes both the image and the text descriptions and calculates a confidence score for each label.
4.  If the confidence score for `"a photo of something else"` is higher, we confidently reject the image.

This process is handled in `app.py` using the `transformers` library from Hugging Face.

### Stage 2: The Breed Classifier (The "Expert")

If, and only if, an image successfully passes the CLIP filter, it is then handed off to our original, highly accurate `EfficientNetV2-S` model, which performs the detailed analysis to determine the specific cattle breed.

## 3. Key Benefits of This Approach

This two-stage architecture makes our application significantly better and more reliable.

*   **Extreme Robustness**: The system no longer breaks on unexpected inputs. Because CLIP was trained on a massive and diverse dataset from the internet, it gracefully handles photos of people, landscapes, objects, and other animals, correctly identifying them as "not cattle".

*   **No Training Required**: We completely eliminated the need for a complex, time-consuming, and error-prone custom training pipeline for the filter. This saves significant development and maintenance time.

*   **State-of-the-Art Performance**: We are leveraging a massive, cutting-edge model from OpenAI, ensuring our filter is highly accurate and reliable without any work on our part.

*   **Efficiency**: The larger, specialized breed model only runs when it's actually needed, saving computational resources on irrelevant images.

## 4. How to Run

This project is a self-contained Flask server.

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    # Ensure numpy is < 2.0 for compatibility
    pip install "numpy<2"
    ```

2.  **Ensure Models are Present**:
    *   The expert model `effnetv2s_best.keras` and `class_names.txt` must be in the root directory.
    *   The CLIP model will be downloaded and cached automatically by the `transformers` library on the first run.

3.  **Run the Server**:
    ```bash
    python app.py
    ```

4.  **Connect Frontends**:
    *   The server will be running at `http://127.0.0.1:5000`.
    *   The prediction endpoint is `POST /predict`.
    *   The server is universally compatible and can handle requests from the brutalist test UI, `orbit-den`, and `swoosh-home`.
