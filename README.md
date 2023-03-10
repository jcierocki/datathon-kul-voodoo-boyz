# KU Leuven LStat Datathon 2023
## Team: Voodoo Boyz

Crew:
- Jakub Cierocki - Capitan, MSc Statistics and Data Science
- Kristaps Greitans, Master of Electromechanical Engineering Technology
- Aleksandra Kotowicz, MSc Statistics and Data Science
- Tristan Vandervelde, MSc Statistics and Data Science

This repository is intended to store all the analysis and codes developed during Datathon 2023.

## Problem description

We were given (by the organisers) a graph databse with various data about artists and artworks:

- artworks with images
- artists with basic bio's
- art movements
- the universities the artists studies at
- the places they were born and died
- artwork types
- syntetic images generated using text-to-speech technology
- etc.

The data was scraped primarly from [wikigallery.org](https://www.wikigallery.org) and included many missing values and data-cleaness issues.

The task was to pursue any analysis and then present it in a way that would impress the jury.

## Data and modelling:

Image data consists of ~ 13k images od size ~ 512x512, being either real or generated, with proportion ~ 1:3 (class imbalance). Among the ~ 10k images of real artwork, we were able to match movements to ~ 4k of them (based on Artwork -- Artist -- Movement relation in the graph). Howewer we were able to provide unique movements only for 2872 images.

We used the data mentioned above to develop 3 different ML models.

1. SVM-based syntetic (generated) image detector:
    - grayscale, 64x64 middle pixels crop
    - 1D Discrete Consine Tranform
    - only high and very low frequencies extracted
    - `log(abs(.))` transformation
    - RBF kernel
    - balanced accuracy out-of-sample: 73%
    - F1 score out-of-sample: 0.83
    - inspiration: see Ricker et al. (2022)
2. Neural-based syntetic (generated) image detector:
    - inital 320x320 middle pixels crop
    - network built-in 225x225 rescale and normalization
    - 2D Fast Fourier Tranform
    - only middle pixels extracted
    - GFNet neural architecture
    - PyTorch backend
    - accuracy out-of-sample: 86%
    - inspiration: see Corvi et al. (2022) and Rao et al. (2021)
3. Neural-based art movement multiclass classifier:
    - rare movements dropped, 51 remained
    - initial 640x640 resize
    - data augmentations used to enlarge train dataset
    - Yolo v8 Nano neural network pre-trained for art movement classification
    - fine-tuned on our dataset
    - TOP5 accuracy: 94%
    - TOP1 accuracy: 40%

Models 1 & 2 were both based on a idea of extracting invisible artifacts specific for a given generative architecture (in our case Stable Diffussion).
  
## References

Corvi, R., Cozzolino, D., Zingarini, G., Poggi, G., Nagano, K., & Verdoliva, L. (2022). On the detection of synthetic images generated by diffusion models.

Rao, Y., Zhao, W., Zhu, Z., Lu, J., & Zhou, J. (2021). Global Filter Networks for Image Classification.

Ricker, J., Damm, S., Holz, T., & Fischer, A. (2022). Towards the Detection of Diffusion Model Deepfakes.

## Usage instructions

To run the main dashboard with visualisations:

`gunicorn src.dashboard:server`

To run 2nd dashboard with Computer Vision models GUI:

`gunicorn src.ml_gui:server`

To create required `venv`, run:

`bash ./setup.sh PATH_TO_VENV`

where `PATH_TO_VENV` needs to be replace with path to desired catalog you want the `venv` be created and all dependencies installed. By aware that `pyenv` is required by the script to work.

To activate `venv`, type:

`source PATH_TO_VENV/bin/activate`

**Be aware that the first dashboard requires Neo4j database to be set up and populate.**

*Neo4j* 5.5 is required. See [link](https://neo4j.com/docs/operations-manual/current/installation/linux/) for installation notes.

Type:

`python src/db_pop.py`

to populate *Neo4j* database. Currently the project is configured to work with default, root *Neo4j* database, which may require to uncomment the line:

`dbms.security.auth_enabled=false`

in */etc/neo4j/neo4j.conf*.
