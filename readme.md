# Feedback Analyzer
## Run on localhost
1. get [Python3](https://www.python.org/downloads/)
2. git clone this project.
3. run `pip3 install -r requirements.txt`
4. download nltk files `py -3 -m nltk.downloader all`
5. run `py -3 linear_regression.py` to generate weights and dictionary for tweeter dataset or run `logistic_regression.py` for news dataset.
6. run `py -3 server.py`
7. goto `http://localhost:8080/index.html`