# CRISPROutcome-ML
# CRISPR Repair Outcome Prediction Pipeline

A machine learning pipeline for predicting DNA repair outcomes after CRISPR-Cas9 editing, inspired by research in repair outcome prediction but independently implemented with custom feature engineering and model architecture.

## Overview

This pipeline predicts four key outcomes of CRISPR-Cas9 editing:
- Fraction of insertions
- Average deletion length  
- Indel diversity
- Fraction of frameshifts

## Input Data Requirements

### Required DataFrame Format
The pipeline expects input data in a pandas DataFrame with the following required columns:
- `Id`: Unique identifier for each sequence
- `GuideSeq`: DNA sequence consisting of 23 base pairs

### Sequence Requirements
The `GuideSeq` column must contain sequences that meet these specifications:
- Total length: 23 base pairs
- First 20 bp: Guide sequence
- Last 3 bp: PAM sequence

Example format:
```python
data = pd.DataFrame({
    'Id': [0, 1, 2],
    'GuideSeq': [
        'CTGCAGGGCTAGTTTCCTATAGG',  # 20bp guide + 3bp PAM
        'GAGATGCGGACCACCCAGCTGGG',
        'GCAAACGGAAGTGCAATTGTCGG'
    ]
})
```

⚠️ **Note**: Sequences not meeting these requirements may cause pipeline errors or produce invalid predictions.

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
