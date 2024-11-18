# CRISPROutcome-ML
# CRISPR Repair Outcome Prediction Pipeline

A machine learning pipeline for predicting DNA repair outcomes after CRISPR-Cas9 editing, inspired by research in repair outcome prediction but independently implemented with custom feature engineering and model architecture.

## Overview

This pipeline predicts four key outcomes of CRISPR-Cas9 editing:
- Fraction of insertions
- Average deletion length  
- Indel diversity
- Fraction of frameshifts

## Features

The pipeline implements:
- Custom feature extraction for guide RNA sequences including:
  - Position-specific nucleotide encoding
  - Cut site context analysis
  - Sequence composition features
- Advanced model architectures:
  - Ensemble LightGBM with bagging
  - Optimized hyperparameters for each target
  - Cross-validation strategy
- Comprehensive feature importance analysis

## Implementation Notes

- Models use optimized hyperparameters for each target
- Feature engineering focuses on biologically relevant aspects
- Ensemble approach helps reduce overfitting

## Acknowledgments

While this implementation is original, it was inspired by research in the field of CRISPR repair outcome prediction, including work by Leenay et al. (2019) and others in the field.

## License

This project is available under the MIT License.
