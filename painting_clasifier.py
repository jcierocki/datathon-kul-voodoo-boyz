from ultralyticsplus import YOLO, postprocess_classify_output
import os

# load model
model = YOLO('keremberke/yolov8m-painting-classification')

# set model parameters
model.overrides['conf'] = 0.4  # model confidence threshold

# set path to image folder
image_folder = './generated_images/'

# loop through images in folder
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg'):
        # set image
        image_path = os.path.join(image_folder, filename)

        # perform inference
        results = model.predict(image_path)

        # observe results
        print(f'{filename}: {results[0].probs}') # e.g. myimage.jpg: [0.1, 0.2, 0.3, 0.4]
        processed_result = postprocess_classify_output(model, result=results[0])
        print(f'{filename}: {processed_result}') # e.g. myimage.jpg: {"cat": 0.4, "dog": 0.6}
