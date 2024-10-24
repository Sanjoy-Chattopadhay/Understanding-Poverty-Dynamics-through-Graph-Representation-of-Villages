import logging
import pandas as pd

logger = logging.getLogger(__name__)

def load_village_data(file_path, num_rows=100):
    logger.info(f'Loading village data from {file_path} with {num_rows} rows')
    try:
        village_data = pd.read_excel(file_path, nrows=num_rows)
        village_data.columns = village_data.columns.str.strip()
        logger.info('Data loaded and cleaned successfully')
        return village_data
    except Exception as e:
        logger.error(f'Failed to load village data: {e}')
        raise
