from flask import Flask
from analyzer import analyze

app = Flask(__name__)
object = analyze()

@app.route('/<text>')
def sentiment_analyze(text):
    result = object.input(text)
    print(text,result)
    return str(result)

app.run()