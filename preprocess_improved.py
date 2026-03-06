"""Backward-compatible entrypoint for the improved preprocessing pipeline."""

from preprocess import preprocess_data


if __name__ == '__main__':
    preprocess_data('cleaned_data.csv')
