# Minimalist GRPO for Molecular Generation

This is a tiny implementation of GRPO inspired by Andreas Koepf's https://github.com/open-thought/tiny-grpo/tree/main.

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To generate molecular structures, you can use the following command:

```bash
python mol_train.py --model_type gru --download_chembl --num_chembl_samples 5000
```
