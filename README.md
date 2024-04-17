# Hallutination-score

Description:
This repository contains code and resources related to hallucination scoring in natural language generation tasks. Hallucination refers to the generation of information that is not present in the input or is not factually accurate. Evaluating and mitigating hallucination is crucial for ensuring the quality and reliability of generated text in various NLP applications.

Contents:

Code: This directory contains scripts and notebooks for computing hallucination scores using different methodologies and datasets.

Datasets: This directory includes datasets used for training and evaluating hallucination scoring models. It may contain both preprocessed datasets and scripts for data preprocessing.

Models: This directory contains pre-trained hallucination scoring models, along with instructions on how to use them for scoring generated text.

Documentation: This directory includes detailed documentation on how to use the code, datasets, and models provided in the repository. It may also contain explanations of the methodologies used for hallucination scoring.

Usage:

To compute hallucination scores for generated text, refer to the documentation for instructions on using the provided code and models.
To train custom hallucination scoring models, utilize the datasets provided in the repository and follow the guidelines outlined in the documentation.


The names of the libraries used in the project are placed in requirements.txt

# the steps required to implement the project

  1. git clone https://github.com/sargismnoyan01/Hallucination-score
  2. %cd Hallucination-score/
  3. pip install -r requirements.txt
  4. python inference.py --model_path facebook/opt-1.3b --task wikibio --only_keyword --use_penalty --add_type --use_idf --       use_entropy --gamma 0.8 --rho 0.01
  5. write the text․
   




