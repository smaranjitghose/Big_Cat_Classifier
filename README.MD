# Big Cat Classifier 🐯

An image classifier built using Vision Transformers that categories images of the big cats into the following classes:

1. Cheetah
2. Jaguar
3. Leopard
4. Lion
5. Tiger

> **Note**:
>
> - Since jaguars and leopards have similar appearances, the model might confuse the two. These [[1](https://www.nationalgeographic.com/animals/article/animals-big-cats-jaguars-leopards)] [[2](https://safarisafricana.com/jaguar-v-leopard/)] two articles throw some light on the difference between the two species.
> - Theoretically the model should be able to accurately identify geographical population variants of each species. However, in practical scenarios this may not be true as during the training phases this was not kept in mind while collecting the dataset.
>   - For example: images of Bengal Tigers, Siberian Tigers, Indochinese Tigers, and Malayan Tigers should be identified as Tigers
> - Lastly, the performance of the model in categorizing certain rare variants in the populations of big cats such as white tigers, snow leopards, or black panther has not been determined exclusively. Although some of the tests performed gave satisfactory results.

## Usage

```python
from PIL import Image
import matplotlib.pyplot as plt
from transformers import ViTFeatureExtractor, ViTForImageClassification

def identify_big_cat(img_path:str)->str:
  """
  Function that reads an image of a big cat (belonging to Panthera family) and returns the corresponding species
  """
  img = Image.open(img_path)
  model_panthera = ViTForImageClassification.from_pretrained("smaranjitghose/big-cat-classifier")
  feature_extractor = ViTFeatureExtractor.from_pretrained('smaranjitghose/big-cat-classifier')
  inputs = feature_extractor(images=img, return_tensors="pt")
  outputs = model_panthera(**inputs)
  logits = outputs.logits
  predicted_class_idx = logits.argmax(-1).item()
  return model_panthera.config.id2label[predicted_class_idx]


 our_big_cat = identify_big_cat("path_of_the_image")
 print(f"Predicted species: {our_big_cat}" )
```

## Reference and Acknowledgement:

[Hugging Pics](https://github.com/nateraw/huggingpics)