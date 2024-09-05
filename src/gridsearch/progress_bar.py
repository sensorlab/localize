from pathlib import Path

import os
from tqdm import tqdm
import time
from multiprocessing import Process
    
import joblib

class GSCVProgressBar():
    """
    Custom progress bar for GridSearchCV that tracks and displays the progress of grid search 
    iterations by monitoring progress and general information files.

    Methods:
    - start(): Waits for progress data to be avaliable and then starts the progress bar
    - stop(): Stop the progress bar process and clean up the progress file
    """
    def __init__(self, progress_file: Path, general_file: Path):
        """
        Initializes the GSCVProgressBar with the paths for the progress and general files.
        
        Args:
        - progress_file (Path): Path to the file where progress information will be stored.
        - general_file: (Path): Path to the file where general grid search information will be stored.
        """
        self.progress_file = progress_file
        self.general_file = general_file
        
        # Create or clear the progress file
        with open(progress_file, 'w') as f:
            f.write("")  

        # Remove the general file if it exists to correctly initilaize the progress bar
        if os.path.exists(general_file):
            os.remove(general_file)
            
            
    def _wait_for_data(self):
        """
        Wait until the general file is created and contains the required information
        about the total number of splits and candidates.
        """
        # Wait for the general file to be created with necessary data
        while not os.path.exists(self.general_file):
            time.sleep(0.1)  # Brief pause before checking again
        
        # Load the general information from the file
        with open(self.general_file, 'rb') as f:
            general_data = joblib.load(f)
            n_splits = general_data['n_splits']
            n_candidates = general_data['n_candidates']
            self.total_iterations = n_splits * n_candidates
        
    def _update_progress_bar(self):
        """
        Continuously updates the progress bar by reading the progress file 
        until all iterations are completed.
        """
        while True:
            # Count the completed iterations by counting lines in the progress file
            with open(self.progress_file, 'r') as f:
                completed_iterations = sum(1 for _ in f)
                
            # Update the progress bar with the number of completed iterations
            self.progress_bar.n = completed_iterations
            self.progress_bar.refresh()
            
            # Break the loop if all iterations are completed
            if completed_iterations >= self.total_iterations:
                break

            time.sleep(0.5)  # Update the progress bar every 0.5 seconds

        self.progress_bar.close() # Close the progress bar after completion
    
    def _progress_bar(self):
        """
        Initializes and updates the progress bar by waiting for the data
        and then continuously updating it until all iterations are completed.
        """
        self._wait_for_data()  # Wait for the necessary data to be available
        
        # Initialize the progress bar with total iterations
        self.progress_bar = tqdm(total=self.total_iterations, desc="Grid Search Progress", unit="iter", dynamic_ncols=True, smoothing=0.1)
        
        self._update_progress_bar()  # Start updating the progress bar
        
        
    def start(self):
        """
        Starts the progress bar updater in a separate process to run concurrently
        with the grid search process.
        """
        self.progress_process = Process(target=self._progress_bar)  # Create a separate process for the progress bar
        self.progress_process.start()  # Start the process
        
    def stop(self):
        """
        Stops the progress bar process and cleans up the progress file.
        """
        self.progress_process.join()  # Wait for the progress bar process to finish
        os.remove(self.progress_file)  # Remove the progress file after completion