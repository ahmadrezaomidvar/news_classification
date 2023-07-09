# News Classification

Developing a news classifier capable of accurately categorising articles into one of seven classes, with a particular focus on **world**, **politics**, and **science**.

## Introduction

In today's fast-paced digital world, staying informed and up-to-date with the latest news is crucial for businesses and individuals alike. Our client, recognising the importance of efficient news categorisation, has requested the development of a news classifier capable of accurately categorising articles into one of seven classes: **world**, **politics**, **science**, **automobile**, **entertainment**, **sports** and **technology**. The primary focus of this project is on the **world**, **politics**, and **science** articles, as these are of the greatest interest to the client. However, the classifier is expected to perform well across all categories.

## Exploratory Data Analysis

An exploratory data analysis was conducted to gain insights into the dataset and identify patterns or trends. This analysis included visualisations of class distribution and word clouds for each category.

![1](images/1.png)
![2](images/2.png)

## Training and Evaluation
For the model training and evaluation, we employed Logistic Regression, Random Forrest, Support Vector Machine and LSTM models. The models' performance were assessed using evaluation metrics such as F1 Score, accuracy, precision, and recall. The models also utilize techniques such as cross-validation, under-sampling, oversampling, and class weight optimisation.

# Installation

## Mambaforge
Download the `mamba` installation file corresponding to your system from here:
https://github.com/conda-forge/miniforge#mambaforge

run the file using:

```bash
bash <FILE_NAME>
```
For example:
```bash
bash Mambaforge-MacOSX-x86_64.sh
```

## Environment installation
After the mamba installation, open a new terminal. Go to this package folder and run:
```bash
make install
```

After the installation, activate the environment:

```bash
mamba activate news
```

## Environment update
If you change any of the environment files, you can update your environment by running:
```bash
make update
```

## Test
To run tests and test coverage, simply run:
```
make test_and_coverage
```

## Experiment
To run the experiments, simply run:
```
make benchmark
```
```
make cross_validation
```
```
make lstm
```