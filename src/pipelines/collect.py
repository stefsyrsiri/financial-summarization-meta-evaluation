from loguru import logger

from src.modules.data_collector import DataCollector


def collect_data():
    # Collect data for destruction and evaluation
    data_collector = DataCollector()
    logger.info("Collecting data.")
    try:
        data_collector.collect_data()
        logger.info("Successfully collected Greek data.")
    except Exception as e:
        logger.exception(f"Failed data collection: {e}")
        return
