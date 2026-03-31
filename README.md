# MLBB Draft Win Predictor

## Overview
MLBB Draft Win Predictor is a machine learning project that predicts the likely winner of a Mobile Legends: Bang Bang match based on the drafted heroes of the Blue and Red teams.

The project uses hero draft composition as input and applies a machine learning classification model to estimate the winning side.

## Problem Statement
Drafting plays a major role in MLBB matches. Players and analysts often want to know whether a given draft composition provides an advantage before the match begins. This project aims to predict the likely winner from the hero lineup alone.

## Features
- Predicts whether Blue side or Red side is more likely to win
- Returns win probabilities for both teams
- Uses a simple and interpretable ML pipeline
- Easy to train and test on custom draft datasets

## Technologies Used
- Python
- pandas
- scikit-learn
- joblib

## Project Structure
```text
mlbb-draft-win-predictor/
│
├── data/
│   └── drafts.csv
├── models/
│   └── mlbb_win_model.pkl
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── main.py
├── requirements.txt
└── README.md
