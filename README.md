# Multilingual Prosody Transfer: Comparing Supervised & Transfer Learning
This repository contains the code for our work titled `Multilingual Prosody Transfer: Comparing Supervised & Transfer Learning`, accepted at the Tiny Track of ICLR'24. 

## Usage
* For loading the dependencies, the given yaml file can be used through the following command:
```bash
conda env create -f Environment.yml
```
* The directory `sft-scripts` contain our scripts for fine-tuning SpeechT5 on the CommonVoice and VoxPopuli datasets.
* `transfer-learning.py` : contains our voice cloning code with a MMS TTS system. It can be used as follows:
```bash
python3 transfer-learning.py \
        --source ___ # Choose the source dataset out of ['voxpopuli', 'commonvoice']
        --language ___ # Specify language codes out of ['fr', 'de', 'es', 'nl', 'hi', 'ta']
```

## Results
<img width="947" alt="Screenshot 2024-10-07 at 2 24 11 PM" src="https://github.com/user-attachments/assets/2f82f642-f126-4bf5-995c-8a00e93708b4">

<img width="1221" alt="Screenshot 2024-10-07 at 2 31 52 PM" src="https://github.com/user-attachments/assets/febaf07c-d6b2-462b-bbfb-75d9397b4daa">


* We additionally provide the script to calculate `Mel Cepstral Distortion (MCD)`.

## Citation
If found helpful, please cite our work using
```bibtex
@inproceedings{
goel2024multilingual,
title={Multilingual Prosody Transfer: Comparing Supervised \& Transfer Learning},
author={Arnav Goel and Medha Hira and Anubha Gupta},
booktitle={The Second Tiny Papers Track at ICLR 2024},
year={2024},
url={https://openreview.net/forum?id=DKF7YCwCmd}
}
```
