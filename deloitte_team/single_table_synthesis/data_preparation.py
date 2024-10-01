from io import StringIO

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin



class CreateFanOutTrainingDataset(BaseEstimator, TransformerMixin):
    def __init__(self, file_paths: list[str]):
        self.file_paths = file_paths
        self.starting_pattern = "BEGIN LAUNDERING ATTEMPT - FAN-OUT"
        self.ending_pattern = "END LAUNDERING ATTEMPT - FAN-OUT"

        self.POSITION_AFTER_DATE_FIELD = 17

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        def _create_fan_out_training_data():
            number_of_fan_out_patterns = 0

            # Initialize variables
            extracted_contents = []
            is_collecting = False

            for file_path in self.file_paths:
                with open(file_path, 'r') as file:
                    for line in file:
                        # Check for the beginning line
                        if line.startswith(self.starting_pattern):
                            is_collecting = True
                            number_of_fan_out_patterns += 1
                            continue  # Skip the beginning line

                        # Check for the ending line
                        if line.startswith(self.ending_pattern):
                            is_collecting = False
                            continue  # Skip the ending line

                        # Collect the content if we're in the relevant section
                        if is_collecting:
                            modified_str = line[:self.POSITION_AFTER_DATE_FIELD] + file_path + line[self.POSITION_AFTER_DATE_FIELD:]
                            extracted_contents.append(modified_str)

                # Join the extracted lines into a single string (or keep as a list)
                result_as_csv_text = ''.join(extracted_contents)

            buffer = StringIO(result_as_csv_text)
            result_as_df = pd.read_csv(buffer)

            return result_as_df, number_of_fan_out_patterns

        result_as_df, number_of_fan_out_patterns = _create_fan_out_training_data()

        print(f"The number_of_fan_out_patterns = {number_of_fan_out_patterns}")
        return result_as_df




if __name__ == '__main__':
    # Create the pipeline
    pipeline = Pipeline([
        ('create_fanout_training_dataset', CreateFanOutTrainingDataset(file_paths=[
            "test/data/HI-Small_Patterns.txt",
            "test/data/HI-Medium_Patterns.txt",
            "test/data/HI-Large_Patterns.txt"
        ]))
    ])

    result_df = pipeline.fit_transform(None)
    print(result_df.head())

