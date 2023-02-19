import cv2
import numpy as np
import os

# Load 1000 images into a list
images = []
for filename in os.listdir("./generated_images"):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join("./generated_images", filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)

            
            
# Denoise each image using filter proposed in [28]
denoised_images = []
for img in images:
    denoised = cv2.fastNlMeansDenoising(img, None, 7, 11, 40)
    denoised_images.append(denoised)

# Take the Fourier transform of each denoised image
fft_images = [np.fft.fft2(img) for img in images]#denoised_images]
fft_images_d = [np.fft.fft2(img) for img in denoised_images]

# Average the Fourier transforms of all images
averaged_denoised = np.mean(fft_images_d, axis=0)
average_fft = np.mean(fft_images, axis=0)
fft_denoised = np.fft.fftshift(averaged_denoised)
fft_shifted = np.fft.fftshift(average_fft)
resulting = fft_denoised - fft_shifted
spectrum = 20 * np.log(np.abs(resulting))

colored_spectrum = cv2.applyColorMap(spectrum.astype(np.uint8), cv2.COLORMAP_JET)
# Display the spectrum
cv2.imwrite("footprint.jpeg", colored_spectrum)
cv2.imshow("Spectrum", spectrum.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
