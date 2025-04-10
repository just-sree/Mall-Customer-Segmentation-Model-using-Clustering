import pandas as pd
from logger import get_logger
from exceptions import DataLoadingError

logger = get_logger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully from {filepath}.")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise DataLoadingError(f"Failed to load data from {filepath}. Error: {e}")
