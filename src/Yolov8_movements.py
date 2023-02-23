from ultralytics import YOLO
import os
from tqdm import tqdm  # pip install tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import splitfolders


class YoloV8Classifier():

    def generateDS(self, original_path, generated_path):
        movement_names = ['Ashcan School', 'Pre-Raphaelite Brotherhood', 'Synthetism', 'Tonalism', 'Symbolism', 'Intimism', 'Newlyn School', 'Regionalism', 'Modernism', 'Les Nabis', 'Academism', 'Divisionism', 'Renaissance', 'Abstract', 'Art Nouveau', 'Romanticism', 'Hudson River School', 'Decadent Movement', 'Munich School', 'Mannerism', 'Rocky Mountain School', 'Bristol School', 'Gothic', 'Neoclassicism', 'Avant-Garde', 'Baroque', 'Venetian School',
                          'Realism', 'Delft School', 'Norwich School', 'Luminism', 'Expressionism', 'Dutch Golden Age', 'Heidelberg School', 'Pointillism', 'Utagawa School', 'Purismo', 'Vienna Secession', 'Suprematism', 'Primitivism', 'Surrealism', 'Aestheticism', 'Early Netherlandish', 'Post-Impressionism', 'Impressionism', 'Futurism', 'Dusseldorf School', 'Berlin Secession', 'Barbizon School', 'Bauhaus', 'Animalier', 'Muralism', 'Orientalism', 'Neo-Impressionism', 'Fauvism']
        self.original_path = original_path
        self.generated_path = generated_path
        os.makedirs(self.generated_path, exist_ok=True)
        for movement_name in movement_names:
            if not os.path.exists(f'{generated_path}/resized/{movement_name}'):
                os.makedirs(f'{generated_path}/resized/{movement_name}')

        for subdir, dirs, files in os.walk(original_path):
            for file in tqdm(files, desc='Loading files from {}'.format(subdir)):
                if not file.startswith('.'):
                    filepath = os.path.join(subdir, file)

                    if filepath.endswith(".jpg") or filepath.endswith(".png"):

                        im = Image.open(filepath)
                        imResize = im.resize((640, 640), Image.ANTIALIAS)
                        imRotate = im.rotate(180)
                        subd = os.path.basename(os.path.dirname(filepath))

                        ressub = os.path.join(subd, file)
                        resized_file_path = os.path.join(
                            generated_path, "/resized")
                        resized_file_path_mov = os.path.join(
                            resized_file_path, ressub)

                        # Save the resized image to the new file path
                        imResize.save(resized_file_path_mov)
                        imRotate.save(
                            resized_file_path_mov[:-4] + '_augmented.png')
        print("Images resized, starting to split")
        output = f"{generated_path}/training_dataset/"
        os.makedirs(output, exist_ok=True)
        splitfolders.ratio(resized_file_path,  # The location of dataset
                           output=output,  # The output location
                           seed=42,  # The number of seed
                           # The ratio of splited dataset
                           ratio=(.75, .15, .1),
                           group_prefix=True,  # If your dataset contains more than one file like ".jpg", ".pdf", etc
                           move=False  # If you choose to move, turn this into True
                           )
        return print("dataset ready at:", output)

    def train(self, architecture: str, pt_weights: str, data_location: str, epochs: int):
        # Load a model
        if architecture == "medium":
            self.model = YOLO("yolov8m-cls.yaml")
        elif architecture == "small":
            self.model = YOLO("yolov8s-cls.yaml")
        elif architecture == "nano":
            self.model = YOLO("yolov8n-cls.yaml")
        elif architecture == "large":
            self.model = YOLO("yolov8l-cls.yaml")
        else:
            return print("wrong name of the model")

        # load a pretrained model (recommended for training)
        self.model = YOLO(pt_weights)

        # Train the model ,optimize=True
        model.train(data=data_location, task="classify",
                    mode="train", epochs=epochs, patience=100, imgsz=640)

    def predict(self, model_path: str, image_path: str):
        # Define the index to name mapping
        index_to_name = {1: 'Abstract', 2: 'Academism', 3: 'Aestheticism', 4: 'Animalier', 5: 'Art Nouveau', 6: 'Ashcan School', 7: 'Avant-Garde', 8: 'Barbizon School', 9: 'Baroque', 10: 'Bauhaus', 11: 'Berlin Secession', 12: 'Bristol School', 13: 'Decadent Movement', 14: 'Divisionism', 15: 'Dusseldorf School', 16: 'Dutch Golden Age', 17: 'Early Netherlandish', 18: 'Expressionism', 19: 'Fauvism', 20: 'Futurism', 21: 'Gothic', 22: 'Heidelberg School', 23: 'Hudson River School', 24: 'Impressionism', 25: 'Luminism', 26: 'Mannerism',
                         27: 'Modernism', 28: 'Munich School', 29: 'Muralism', 30: 'Neo-Impressionism', 31: 'Neoclassicism', 32: 'Newlyn School', 33: 'Norwich School', 34: 'Orientalism', 35: 'Pointillism', 36: 'Post-Impressionism', 37: 'Pre-Raphaelite Brotherhood', 38: 'Primitivism', 39: 'Purismo', 40: 'Realism', 41: 'Regionalism', 42: 'Renaissance', 43: 'Rocky Mountain School', 44: 'Romanticism', 45: 'Suprematism', 46: 'Surrealism', 47: 'Symbolism', 48: 'Synthetism', 49: 'Tonalism', 50: 'Venetian School', 51: 'Vienna Secession'}

        # Load the model
        # self.model = YOLO("yolov8m-cls.yaml")  # build model
        self.model = YOLO(model_path)  # path to the best weigths ends with .pt
        # Set model confidence threshold
        self.model.overrides['conf'] = 0.25
        # Set path to image folder
        self.image_folder = image_path
        # Create a list to store predictions for each image
        predictions = []
        # Loop through images in folder
        for filename in os.listdir(self.image_folder):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                # Set image
                image_path = os.path.join(self.image_folder, filename)
                im = Image.open(image_path)
                draw = ImageDraw.Draw(im)
                font = ImageFont.load_default()
                imResize = im.resize((640, 640), Image.LANCZOS)
                # Perform inference
                results = self.model.predict(imResize, show=False)
                # print(results)
                results = results[0].to("cpu")
                # print(results)
                # pass
                results, index = results.probs.topk(3)

                # Get predicted class names and their probabilities
                index_names = [index_to_name[i] for i in index.tolist()]
                probs = results.tolist()
                # draw results
                draw.rectangle([0, 0, 200, 200], outline="red")
                draw.text((5, 5), (index_names, probs),
                          font=font, align="left")
                # Create a dictionary to store the predictions for this image
                im.save("predictions.png")
                prediction = {"filename": filename, "predictions": []}

                # Loop through the predicted class names and their probabilities and add them to the dictionary
                for name, prob in zip(index_names, probs):
                    prediction["predictions"].append(
                        {"movement": name, "probability": 100*prob})

                # Add the dictionary to the list of predictions
                predictions.append(prediction)

        # Print the list of predictions
        # print(predictions)
        return predictions


if __name__ == "__main__":
    model = YoloV8Classifier()
    results = model.predict("./weights/best.pt",
                            "original_images_movement/Abstract")

    print(results[0])
