from pathlib import Path
import warnings
import os
import gc

from unittest.mock import patch
import joblib

from .progress_bar import GSCVProgressBar
from sklearn.model_selection._validation import _fit_and_score as original_fit_and_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def make_prediction_saver_scorer(candidate_idx, split_idx, tmp_dir_path):
    # Adds context to the scorer
    def prediction_saver_metric(y_true, y_pred):
        filename = os.path.join(tmp_dir_path, f"pred-{candidate_idx}-{split_idx}.pkl")
        joblib.dump({'y_true': y_true, 'y_pred': y_pred}, filename)
        # Return 0 to not affect the scoring
        return 0
    return make_scorer(prediction_saver_metric, greater_is_better = False)

class GridSearchCVWithStoredModels(GridSearchCV):
    """
    A custom extension of sklearn's GridSearchCV that stores the models from each split
    and candidate during the grid search process. Additionally, it can show a progress
    bar during the fitting process.
    """

    def fit(self, *args, tmp_dir_path:Path, show_progress_bar: bool = False, **kwargs):
        """
        A wrapper for the fit function that patches the default _fit_and_score function with a custom wrapper for it,
        that handles the model saving and progress tracking.

        Args:
        - tmp_dir_path (Path): Path to the directory where temporary files (models, progress information) will be saved.
        - show_progress_bar (bool) optional: Whether or not to display a progress bar during the fitting process.
        """
        progress_file = os.path.join(tmp_dir_path, "progress.txt")
        general_file = os.path.join(tmp_dir_path, "general.pkl")

        # Initialize the progress bar if requested
        if show_progress_bar:
            progress_bar = GSCVProgressBar(progress_file = progress_file, general_file = general_file)
            progress_bar.start()

        def fit_and_score_wrapper(*args, **kwargs):

             # Ensure the estimator is returned in the results
            kwargs['return_estimator'] = True

             # Extract the split and candidate indices
            split_idx = kwargs["split_progress"][0]
            candidate_idx = kwargs["candidate_progress"][0]

            # Modify the scorer to include our custom scorer
            original_scorer = kwargs['scorer']

            # Make a copy to avoid modifying the original scorer
            if isinstance(original_scorer, dict):
                # The scorer is already a dictionary of scorers
                modified_scorer = original_scorer.copy()
            else:
                # The scorer is a single callable; convert it to a dict
                modified_scorer = {'score': original_scorer}

            # Create our custom metric function
            prediction_saver_scorer = make_prediction_saver_scorer(
                candidate_idx, split_idx, tmp_dir_path
            )

            # Add the custom scorer to the scorer dict
            modified_scorer['prediction_saver'] = prediction_saver_scorer

            # Replace the scorer in kwargs with our modified scorer
            kwargs['scorer'] = modified_scorer

            results = original_fit_and_score(*args, **kwargs)



            # Save the estimator to a file for this candidate and split
            joblib.dump(results["estimator"], os.path.join(tmp_dir_path, f"est-{candidate_idx}-{split_idx}.pkl"))

            # Delete the setimator so that it's not retained in memory
            del results["estimator"]

            # Remove the custom scorer results from the output to avoid confusion
            keys_to_remove = [key for key in results if 'prediction_saver' in key]
            for key in keys_to_remove:
                del results[key]

            if candidate_idx == 0:
                joblib.dump(kwargs["test"], os.path.join(tmp_dir_path, f"split-{split_idx}"))

                if split_idx == 0:
                    joblib.dump({"n_splits":kwargs["split_progress"][1],
                                "n_candidates":kwargs["candidate_progress"][1]}, general_file)
            # Update progress if the progress bar is being shown
            if show_progress_bar:
                # Record progress in the file
                with open(progress_file, 'a') as f:
                    f.write("1\n")

            gc.collect()
            return results

        # Patch the _fit_and_score function to include the custom wrapper
        with patch("sklearn.model_selection._search._fit_and_score", fit_and_score_wrapper):
            # Suppress the specific UserWarning from joblib's loky backend
            warnings.filterwarnings(
                "ignore",
                message="A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.",
                category=UserWarning,
                module="joblib.externals.loky.process_executor")

            # Call the origina lfit method
            super().fit(*args, **kwargs)
        if show_progress_bar:
            # Stop the progress bar if it was started
            progress_bar.stop()

        gc.collect()
