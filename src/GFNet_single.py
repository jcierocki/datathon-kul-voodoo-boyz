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

    def predict(self, model, image: Image):
        # Define the index to name mapping
        index_to_name = {0: 'Fake', 1: 'Real'}

        # self.image_folder = image_path
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # model.to(device)
        # dataset_val, _ = build_dataset(is_train=False, args=("../../Dataset_fake_split", "gfnet-xs", './logs/gfnet-xs/checkpoint_best.pth'))
        trans = transforms.Compose([transforms.Resize(225, interpolation=4),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))])

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
                prediction = {"filename": ".", "predictions": []}

                # Loop through the predicted class names and their probabilities and add them to the dictionary
                for name, prob in zip(index_names, probs):
                    prediction["predictions"].append(
                        {"type": name})

                # Add the dictionary to the list of predictions

            # print(predictions)
        return prediction


if __name__ == "__main__":
    model = GFNetClassifier()
    results = model.predict("./weights/GFNet.pth",
                            "./generated_images/A painting of 6th Sokol Festival. 1912 in the style of Alphonse Maria Mucha_0.jpg")

    print(results)
