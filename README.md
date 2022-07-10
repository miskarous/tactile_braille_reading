Braille letters
=============

Braille is a tactile writing system used by people who are visually impaired. These characters have rectangular blocks called *cells* that have tiny bumps called *raised dots*. The number and arrangement of these dots distinguish one character from another. For more details and background information see [here](https://en.wikipedia.org/wiki/Braille).

![brialle_system_english](https://user-images.githubusercontent.com/60852381/120632860-bb6c9e00-c469-11eb-8b33-47df012f76b0.jpg)

# The Dataset
The dataset is composed of different levels of complexity from single letters to words. The 27 letters (Space + A - Z) have been recorded using the iCub fingertip sliding over 3d printed stimuli. For that, the fingertip was mounted on a 3-axis robot (omega.3, [forcedimensions](https://www.forcedimension.com/products/omega)) and moved over single braille letters 50 times each with similar velocity (0.01 m/s) at a sampling frequency of 40Hz. The data is converted into spike trains afterward. 
Delta coding is used for the conversion. No additional noise is added because the analog recordings already contain sensor noise. Binary events ('ON'/'OFF') are created when a predefined threshold is reached followed by a refractory period. At the end of the refractory period, change is accumulated again, until the threshold is reached and a new event is elicit. Thresholds and refractory period are (0.5 for ON and OFF) and (0.0025 sec) respectively. The recordings of the single letters spike trains are combined to compose words.

Experimental Setup | Encoding Scheme
:------------:|:------------:
![experiantal_setup](https://github.com/event-driven-robotics/tactile_braille_reading/blob/main/assets/acquisition_setup.JPG) | ![encoding_scheme](https://github.com/event-driven-robotics/tactile_braille_reading/blob/main/assets/figure_encoding-reconstruct.JPG)

Scanning | Sample-based | Event-based 
:------------:|:------------:|:------------:
![scanning](https://github.com/event-driven-robotics/tactile_braille_reading/blob/main/assets/pipeline1.gif) | ![sample_based](https://github.com/event-driven-robotics/tactile_braille_reading/blob/main/assets/pipeline2.gif) | ![event_based](https://github.com/event-driven-robotics/tactile_braille_reading/blob/main/assets/pipeline3.gif)

# How-to
1. Install [python](https://www.python.org/) and [pytorch](https://pytorch.org/)
2. Download the [dataset](https://zenodo.org/record/6556273) from Zenodo
3. Extract the files and add them in the main folder of this repository
4. Download the [Izhikevich Encoded data](https://drive.google.com/file/d/1dBH9F0OoTCknMKkNM15z5nytaI0Szhrt/view?usp=sharing) and put it in a 'data' folder in the 'notebooks' directory
5. Run the jupiter notebook for the [recurrent SNN using the Izhikevich Neuron Model](https://github.com/miskarous/tactile_braille_reading/blob/main/notebooks/braille_reading_rsnn_izhi_Loihi.ipynb)

