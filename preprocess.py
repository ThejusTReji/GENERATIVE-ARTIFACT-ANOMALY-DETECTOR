import cv2
import os
import numpy as np

real_input = "dataset/train/real"
real_output = "processed/real"

fake_input = "dataset/train/fake"
fake_output = "processed/fake"

os.makedirs(real_output, exist_ok=True)
os.makedirs(fake_output, exist_ok=True)

def process_images(input_folder, output_folder):

    files = os.listdir(input_folder)

    for file in files:

        path = os.path.join(input_folder, file)

        image = cv2.imread(path)

        if image is None:
            continue

        image = cv2.resize(image, (224,224))

        image = image / 255.0

        save_path = os.path.join(output_folder, file)

        cv2.imwrite(save_path,
                    (image * 255).astype(np.uint8))

    print("Done:", output_folder)

process_images(real_input, real_output)
process_images(fake_input, fake_output)