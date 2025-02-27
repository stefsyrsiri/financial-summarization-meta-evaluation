# Financial Narrative Summarization Evaluation (In-progress)

## Overview
In this project we evaluate automatic summarization evaluation methods on the summaries of Greek annual reports.

## Project structure
```sh
thesis/
│── src/
│   ├── main.py               # Main script
│   ├── data_collector.py       # Data handling class
│   ├── summary_destructor.py # Summary destruction class
│   ├── summary_generator.py  # Noisy summary generator class
│   ├── summary_evaluator.py  # Evaluation class
│── data/
│── README.md
│── requirements.txt
│── .gitignore
```

## Installation
1. Clone the repository:
```sh
git clone https://github.com/stefsyrsiri/thesis.git
cd thesis
```

2. Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage
Run the script:
```sh
python src/main.py
```
