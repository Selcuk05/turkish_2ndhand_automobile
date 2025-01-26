import torch
import cv2
from src.classifier.model import HierarchicalCNN
from src.classifier.data import get_transforms


class CarClassifier:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        self.transform = get_transforms(is_training=False)

        # Load class mappings from the model
        self.brand_classes = self.model.brand_classes
        self.series_classes = self.model.series_classes
        self.model_classes = self.model.model_classes

    def preprocess_image(self, image_path):
        """Preprocess a single image for inference"""
        # Read and convert image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed["image"]

        # Add batch dimension
        return image_tensor.unsqueeze(0)

    @torch.no_grad()
    def predict(self, image_path, return_probs=False):
        """
        Predict brand, series, and model for a single image

        Args:
            image_path (str): Path to the image file
            return_probs (bool): Whether to return prediction probabilities

        Returns:
            dict: Predictions for brand, series, and model
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)

        # Get model predictions
        outputs = self.model(image_tensor)

        # Get predicted classes
        predictions = {}
        probabilities = {}

        for key in ["brand", "series", "model"]:
            probs = torch.softmax(outputs[key], dim=1)[0]
            pred_idx = probs.argmax().item()

            # Get class name based on the key
            if key == "brand":
                pred_class = self.brand_classes[pred_idx]
            elif key == "series":
                pred_class = self.series_classes[pred_idx]
            else:
                pred_class = self.model_classes[pred_idx]

            predictions[key] = pred_class
            if return_probs:
                probabilities[key] = probs[pred_idx].item()

        if return_probs:
            return predictions, probabilities
        return predictions

    @torch.no_grad()
    def predict_batch(self, image_paths, batch_size=32):
        """
        Predict brand, series, and model for a batch of images

        Args:
            image_paths (list): List of image file paths
            batch_size (int): Batch size for processing

        Returns:
            list: List of prediction dictionaries
        """
        all_predictions = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_tensors = torch.stack(
                [self.preprocess_image(path)[0] for path in batch_paths]
            ).to(self.device)

            # Get model predictions
            outputs = self.model(batch_tensors)

            # Process each image in the batch
            for j in range(len(batch_paths)):
                predictions = {}
                for key in ["brand", "series", "model"]:
                    probs = torch.softmax(outputs[key][j], dim=0)
                    pred_idx = probs.argmax().item()

                    # Get class name based on the key
                    if key == "brand":
                        pred_class = self.brand_classes[pred_idx]
                    elif key == "series":
                        pred_class = self.series_classes[pred_idx]
                    else:
                        pred_class = self.model_classes[pred_idx]

                    predictions[key] = pred_class

                all_predictions.append(predictions)

        return all_predictions


# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = CarClassifier("path/to/saved/model.pth")

    # Single image prediction
    image_path = "data/processed_images/car_20250126_131100_offset_0_idx_0.jpg"
    predictions = classifier.predict(image_path)
    print(f"Predictions for {image_path}:")
    print(f"Brand: {predictions['brand']}")
    print(f"Series: {predictions['series']}")
    print(f"Model: {predictions['model']}")

    # Batch prediction
    image_paths = [
        "data/processed_images/car_20250126_131100_offset_50_idx_4.jpg",
        "data/processed_images/car_20250126_131100_offset_50_idx_5.jpg",
        "data/processed_images/car_20250126_131100_offset_50_idx_6.jpg",
    ]
    batch_predictions = classifier.predict_batch(image_paths)
    for path, preds in zip(image_paths, batch_predictions):
        print(f"\nPredictions for {path}:")
        print(f"Brand: {preds['brand']}")
        print(f"Series: {preds['series']}")
        print(f"Model: {preds['model']}")
