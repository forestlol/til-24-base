from typing import List, Tuple
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io

class VLMManager:
    def __init__(self):
        # Initialize the model and processor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def identify(self, image: bytes, caption: str) -> Tuple[List[int], str]:
        # Convert bytes to PIL Image
        pil_image = Image.open(io.BytesIO(image))
        
        # Prepare inputs for the model
        inputs = self.processor(text=[caption], images=pil_image, return_tensors="pt", padding=True)

        # Perform inference
        outputs = self.model(**inputs)

        # Get logits
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)  # Softmax to get probabilities
        
        # For simplicity, let's assume the whole image is the bounding box
        bounding_box = [0, 0, pil_image.width, pil_image.height]

        # Get the predicted label
        pred_label_index = torch.argmax(probs).item()
        pred_label = self.processor.tokenizer.decode(pred_label_index)

        return bounding_box, pred_label

# Example usage
if __name__ == "__main__":
    vlm_manager = VLMManager()
    with open("example_image.jpg", "rb") as img_file:
        image_bytes = img_file.read()
    caption = "a photo of a cat"
    bbox, label = vlm_manager.identify(image_bytes, caption)
    print(f"Bounding Box: {bbox}, Label: {label}")
