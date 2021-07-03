from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification


def classifier(img_path: str) -> str:
    """
    Function that reads an image of a big cat (belonging to Panthera family) and returns the corresponding species
    """
    img = Image.open(img_path)
    model_panthera = ViTForImageClassification.from_pretrained(
        "smaranjitghose/big-cat-classifier"
    )
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "smaranjitghose/big-cat-classifier"
    )
    inputs = feature_extractor(images=img, return_tensors="pt")
    outputs = model_panthera(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model_panthera.config.id2label[predicted_class_idx]
