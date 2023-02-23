import gfnet
import numpy as np

import torch
import os
import itertools
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from torchvision.datasets.folder import default_loader


class GFNetClassifier():

    def predict(self, model_path: str, image_path: str):
        # Define the index to name mapping
        index_to_name = {0: 'Fake', 1: 'Real'}

        self.image_folder = image_path
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = torch.load(model_path)
        model.eval()
        # model.to(device)
        # dataset_val, _ = build_dataset(is_train=False, args=("../../Dataset_fake_split", "gfnet-xs", './logs/gfnet-xs/checkpoint_best.pth'))
        trans = transforms.Compose([transforms.Resize(225, interpolation=4),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))])
        # Create a list to store predictions for each image
        predictions = []
        # Loop through images in folder
        for file in os.listdir(self.image_folder):
            file_path = os.path.join(self.image_folder, file)
            # Get the width and height of the image
            image = Image.open(file_path).convert("RGB")
            width, height = image.size

            # Set the size of the crop
            crop_size = 320

            # Calculate the left, upper, right, and lower coordinates of the crop box
            left = (width - crop_size) / 2
            upper = (height - crop_size) / 2
            right = (width + crop_size) / 2
            lower = (height + crop_size) / 2

            # Crop the center of the image
            image = image.crop((left, upper, right, lower))
            test = trans(image)
            test = test[None, :, :, :]
            loader = torch.utils.data.DataLoader(
                test, batch_size=1, num_workers=4)
            with torch.no_grad():
                for batch in loader:
                    output = model(batch)
                    results, index = output.topk(1)

                    # print(index.tolist())
                    index_flat = list(
                        itertools.chain.from_iterable(index.tolist()))
                    # Store the predictions in the dictionary
                    index_names = [
                        index_to_name[i] for i in index_flat]
                    probs = results.tolist()

                    # Create a dictionary to store the predictions for this image
                    prediction = {"filename": file, "predictions": []}

                    # Loop through the predicted class names and their probabilities and add them to the dictionary
                    for name, prob in zip(index_names, probs):
                        prediction["predictions"].append(
                            {"movement": name})

                    # Add the dictionary to the list of predictions
                    predictions.append(prediction)

                # print(predictions)
        return predictions


if __name__ == "__main__":
    model = GFNetClassifier()
    results = model.predict("./weights/GFNet.pth",
                            "./generated_images/")

    print(results[1])
