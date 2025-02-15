import os
import torch
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
from src.classifier.model import DynamicHierarchicalClassifier
from src.classifier.hierarchy_builder import VehicleHierarchyBuilder
from src.pricing.predictor import VehiclePricePredictor
from ultralytics import YOLO
import logging


class CarClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Classifier")
        self.root.geometry("1600x800")
        self.setup_model()
        self.create_widgets()
        self.price_predictor = VehiclePricePredictor(
            "data/turkish_2ndhand_automobile_processed.csv"
        )
        self.price_predictor.load_model()
        self.current_image = None

        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        self.vehicle_segmenter = YOLO("yolov8x-seg.pt", verbose=False)
        self.vehicle_segmenter.to("cuda" if torch.cuda.is_available() else "cpu")

    def setup_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data/turkish_2ndhand_automobile_processed.csv",
        )
        self.hierarchy = VehicleHierarchyBuilder(csv_path).build_hierarchy()
        self.model = DynamicHierarchicalClassifier(self.hierarchy)

        checkpoint_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model = self.model.to(self.device).eval()
        else:
            raise FileNotFoundError("Model weights not found")

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, padx=10)

        self.image_label = ttk.Label(left_frame)
        self.image_label.grid(row=0, column=0, columnspan=4, pady=10)

        ttk.Button(left_frame, text="Select Image", command=self.select_image).grid(
            row=1, column=0, columnspan=4, pady=10
        )

        result_frame = ttk.LabelFrame(
            left_frame, text="Classification Results", padding="10"
        )
        result_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(result_frame, text="Marka (Brand):").grid(
            row=0, column=0, sticky=tk.W
        )
        self.brand_label = ttk.Label(result_frame, text="")
        self.brand_label.grid(row=0, column=1, sticky=tk.W)
        self.brand_confidence = ttk.Label(result_frame, text="")
        self.brand_confidence.grid(row=0, column=2, padx=10, sticky=tk.W)
        self.brand_uncertainty = ttk.Label(result_frame, text="")
        self.brand_uncertainty.grid(row=0, column=3, padx=10, sticky=tk.W)

        ttk.Label(result_frame, text="Seri (Series):").grid(
            row=1, column=0, sticky=tk.W
        )
        self.model_label = ttk.Label(result_frame, text="")
        self.model_label.grid(row=1, column=1, sticky=tk.W)
        self.model_confidence = ttk.Label(result_frame, text="")
        self.model_confidence.grid(row=1, column=2, padx=10, sticky=tk.W)
        self.model_uncertainty = ttk.Label(result_frame, text="")
        self.model_uncertainty.grid(row=1, column=3, padx=10, sticky=tk.W)

        ttk.Label(result_frame, text="Model (Trim):").grid(row=2, column=0, sticky=tk.W)
        self.trim_label = ttk.Label(result_frame, text="")
        self.trim_label.grid(row=2, column=1, sticky=tk.W)
        self.trim_confidence = ttk.Label(result_frame, text="")
        self.trim_confidence.grid(row=2, column=2, padx=10, sticky=tk.W)
        self.trim_uncertainty = ttk.Label(result_frame, text="")
        self.trim_uncertainty.grid(row=2, column=3, padx=10, sticky=tk.W)

        ttk.Label(result_frame, text="Year:").grid(row=3, column=0, sticky=tk.W)
        self.year_entry = ttk.Entry(result_frame)
        self.year_entry.grid(row=3, column=1, sticky=tk.W)

        ttk.Label(result_frame, text="Kilometers:").grid(row=4, column=0, sticky=tk.W)
        self.km_entry = ttk.Entry(result_frame)
        self.km_entry.grid(row=4, column=1, sticky=tk.W)

        ttk.Button(
            result_frame, text="Estimate Price", command=self.estimate_price
        ).grid(row=5, column=0, columnspan=2, pady=10)

        self.price_label = ttk.Label(
            result_frame, text="", font=("Helvetica", 12, "bold")
        )
        self.price_label.grid(row=6, column=0, columnspan=4, pady=10)

        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, padx=10)

        attention_frame = ttk.LabelFrame(
            right_frame, text="Attention Maps", padding="10"
        )
        attention_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.attention_canvas = None

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.display_image(file_path)
            self.classify_image(file_path)

    def display_image(self, path):
        image = Image.open(path).convert("RGB")
        image.thumbnail((400, 300), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def remove_background(self, image):
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        results = self.vehicle_segmenter(cv_image, classes=[2, 3, 5, 7], verbose=False)

        if len(results) == 0 or not results[0].masks:
            return image

        combined_mask = None
        try:
            for mask in results[0].masks:
                mask_np = mask.data.cpu().numpy()
                if mask_np.size == 0:
                    continue

                if combined_mask is None:
                    combined_mask = mask_np
                else:
                    combined_mask = np.logical_or(combined_mask, mask_np)

            if combined_mask is None or combined_mask.size == 0:
                return image

            if len(combined_mask.shape) > 2:
                combined_mask = combined_mask.squeeze()

            combined_mask = cv2.resize(
                combined_mask.astype(float), (cv_image.shape[1], cv_image.shape[0])
            )
            combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
            mask_3channel = np.stack([combined_mask] * 3, axis=-1)
            white_bg = np.ones_like(cv_image) * 255
            result = cv_image * mask_3channel + white_bg * (1 - mask_3channel)

            return Image.fromarray(
                cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)
            )
        except Exception as e:
            print(f"Error in mask processing: {str(e)}")
            return image

    def classify_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")

            processed_image = self.remove_background(image)
            if processed_image is image:
                raise ValueError("No vehicle detected in the image")

            image_tensor = self.transform(processed_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                predictions = self.model.predict_with_uncertainty(
                    image_tensor, num_samples=10
                )

                brand_mean = predictions["brand"]["mean"].squeeze()
                if brand_mean.dim() == 0:
                    raise ValueError("Model failed to generate valid predictions")

                brand_var = predictions["brand"]["variance"].squeeze()
                brand_probs = F.softmax(brand_mean / 0.1, dim=0)
                brand_idx = torch.argmax(brand_probs).item()
                brand_name = self.hierarchy["levels"][0]["classes"][brand_idx]
                brand_conf = brand_probs[brand_idx].item() * 100
                brand_uncertainty = torch.sqrt(brand_var[brand_idx]).item() * 100

                self.current_brand = brand_name

                model_mean = predictions["model"]["mean"][brand_name].squeeze()
                if model_mean.dim() == 0:
                    raise ValueError("Model failed to generate valid predictions")

                model_var = predictions["model"]["variance"][brand_name].squeeze()
                model_probs = F.softmax(model_mean / 0.1, dim=0)
                model_idx = torch.argmax(model_probs).item()
                model_name = self.hierarchy["levels"][1]["classes"][brand_name][
                    model_idx
                ]
                model_conf = model_probs[model_idx].item() * 100
                model_uncertainty = torch.sqrt(model_var[model_idx]).item() * 100

                self.current_model = model_name

                trim_key = f"{brand_name}_{model_name}"
                trim_mean = predictions["trim"]["mean"][trim_key].squeeze()
                if trim_mean.dim() == 0:
                    raise ValueError("Model failed to generate valid predictions")

                trim_var = predictions["trim"]["variance"][trim_key].squeeze()
                trim_probs = F.softmax(trim_mean / 0.1, dim=0)
                trim_idx = torch.argmax(trim_probs).item()
                trim_name = self.hierarchy["levels"][2]["classes"][trim_key][trim_idx]
                trim_conf = trim_probs[trim_idx].item() * 100
                trim_uncertainty = torch.sqrt(trim_var[trim_idx]).item() * 100

                self.current_trim = trim_name

                self.brand_label.config(text=brand_name)
                self.brand_confidence.config(text=f"Conf: {brand_conf:.1f}%")
                self.brand_uncertainty.config(text=f"Unc: {brand_uncertainty:.1f}%")

                self.model_label.config(text=model_name)
                self.model_confidence.config(text=f"Conf: {model_conf:.1f}%")
                self.model_uncertainty.config(text=f"Unc: {model_uncertainty:.1f}%")

                self.trim_label.config(text=trim_name)
                self.trim_confidence.config(text=f"Conf: {trim_conf:.1f}%")
                self.trim_uncertainty.config(text=f"Unc: {trim_uncertainty:.1f}%")

                self._update_label_colors(
                    self.brand_label,
                    self.brand_confidence,
                    self.brand_uncertainty,
                    brand_conf,
                    brand_uncertainty,
                )
                self._update_label_colors(
                    self.model_label,
                    self.model_confidence,
                    self.model_uncertainty,
                    model_conf,
                    model_uncertainty,
                )
                self._update_label_colors(
                    self.trim_label,
                    self.trim_confidence,
                    self.trim_uncertainty,
                    trim_conf,
                    trim_uncertainty,
                )

                attention_maps = self.generate_attention_maps(image_tensor, predictions)
                self.display_attention_maps(image_path, attention_maps)

        except Exception as e:
            print(f"Error during classification: {str(e)}")
            self.brand_label.config(text="Error")
            self.model_label.config(text="Error")
            self.trim_label.config(text="Error")
            self.brand_confidence.config(text="")
            self.model_confidence.config(text="")
            self.trim_confidence.config(text="")

    def _update_label_colors(
        self, label, conf_label, unc_label, confidence, uncertainty
    ):
        """Update label colors based on confidence and uncertainty"""
        if confidence > 90 and uncertainty < 5:
            color = "green"
        elif confidence > 70 and uncertainty < 15:
            color = "orange"
        else:
            color = "red"

        label.config(foreground=color)
        conf_label.config(foreground=color)
        unc_label.config(foreground=color)

    def estimate_price(self):
        try:
            year = int(self.year_entry.get())
            km = float(self.km_entry.get().replace(".", ""))

            price = self.price_predictor.predict(
                self.current_brand, self.current_model, self.current_trim, year, km
            )

            self.price_label.config(
                text=f"Estimated Price: {price:,.0f} TL", foreground="green"
            )
        except Exception as e:
            self.price_label.config(text=f"Error: {str(e)}", foreground="red")

    def generate_attention_maps(self, image_tensor, predictions):
        """Generate attention maps for brand, model, and trim predictions"""
        attention_maps = {}

        image_tensor = image_tensor.clone().detach().requires_grad_(True)

        torch.set_grad_enabled(True)
        self.model.train()

        try:
            brand_logits = self.model.forward_brand(image_tensor)
            brand_idx = predictions["brand"]["mean"].argmax(dim=1)[0]
            brand_score = brand_logits[0, brand_idx]

            brand_score.backward(retain_graph=True)
            brand_grads = image_tensor.grad.abs()
            brand_attention = brand_grads.mean(dim=1).squeeze().cpu().numpy()
            attention_maps["brand"] = brand_attention

            image_tensor.grad = None

            brand_name = self.current_brand
            model_logits = self.model.forward_model(image_tensor, brand_name)
            model_idx = predictions["model"]["mean"][brand_name].argmax(dim=1)[0]
            model_score = model_logits[0, model_idx]

            model_score.backward(retain_graph=True)
            model_grads = image_tensor.grad.abs()
            model_attention = model_grads.mean(dim=1).squeeze().cpu().numpy()
            attention_maps["model"] = model_attention

            image_tensor.grad = None

            trim_key = f"{self.current_brand}_{self.current_model}"
            trim_logits = self.model.forward_trim(image_tensor, trim_key)
            trim_idx = predictions["trim"]["mean"][trim_key].argmax(dim=1)[0]
            trim_score = trim_logits[0, trim_idx]

            trim_score.backward()
            trim_grads = image_tensor.grad.abs()
            trim_attention = trim_grads.mean(dim=1).squeeze().cpu().numpy()
            attention_maps["trim"] = trim_attention

        finally:
            torch.set_grad_enabled(False)
            self.model.eval()

        return attention_maps

    def display_attention_maps(self, image_path, attention_maps):
        fig, axs = plt.subplots(2, 3, figsize=(10, 6))
        fig.suptitle("Attention Maps", fontsize=12)

        orig_img = cv2.imread(image_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        bg_removed_img = self.remove_background(Image.fromarray(orig_img))
        bg_removed_img = np.array(bg_removed_img)

        axs[0, 0].imshow(orig_img)
        axs[0, 0].set_title("Original", fontsize=9)
        axs[0, 0].axis("off")

        axs[0, 1].imshow(bg_removed_img)
        axs[0, 1].set_title("Background Removed", fontsize=9)
        axs[0, 1].axis("off")

        axs[0, 2].axis("off")

        titles = ["Brand Focus", "Model Focus", "Trim Focus"]
        positions = [(1, 0), (1, 1), (1, 2)]
        levels = ["brand", "model", "trim"]

        for title, pos, level in zip(titles, positions, levels):
            attention = attention_maps[level]

            attention = cv2.resize(
                attention, (bg_removed_img.shape[1], bg_removed_img.shape[0])
            )
            attention = cv2.GaussianBlur(attention, (5, 5), 0)
            attention = (attention - attention.min()) / (
                attention.max() - attention.min() + 1e-8
            )

            if level == "brand":
                threshold = np.percentile(attention, 85)
                attention[attention < threshold] = 0
            elif level == "model":
                threshold = np.percentile(attention, 80)
                attention[attention < threshold] *= 0.2
            else:  # trim
                threshold = np.percentile(attention, 75)
                attention[attention < threshold] *= 0.3

            # Create a binary mask for the high attention regions
            attention_mask = (attention > threshold).astype(float)
            attention_mask = cv2.GaussianBlur(attention_mask, (3, 3), 0)

            highlighted = bg_removed_img.copy()
            highlight_color = np.array([255, 255, 0], dtype=np.float32)
            for c in range(3):
                highlighted[:, :, c] = (
                    bg_removed_img[:, :, c] * (1 - attention_mask * 0.3)
                    + highlight_color[c] * attention_mask * 0.3
                )

            heatmap = cv2.applyColorMap(np.uint8(255 * attention), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            blend_factor = attention_mask * 0.7
            overlay = bg_removed_img.copy()
            for c in range(3):
                overlay[:, :, c] = (
                    highlighted[:, :, c] * (1 - blend_factor)
                    + heatmap[:, :, c] * blend_factor
                )

            edges = cv2.Canny(np.uint8(attention_mask * 255), 100, 200)
            overlay[edges > 0] = [255, 255, 255]

            axs[pos[0], pos[1]].imshow(overlay)
            axs[pos[0], pos[1]].set_title(
                f"{title}\n({int(threshold * 100)}%)", fontsize=9
            )
            axs[pos[0], pos[1]].axis("off")

        plt.tight_layout(pad=1.0)

        if self.attention_canvas is not None:
            self.attention_canvas.get_tk_widget().destroy()

        self.attention_canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.attention_canvas.draw()
        self.attention_canvas.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)


def main():
    root = tk.Tk()
    CarClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
