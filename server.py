from flask import Flask
from analyzer import analyze
import os

host = 'localhost'
if "IP" in os.environ.keys():
    host = os.environ["IP"]

port = 8080
if "PORT" in os.environ.keys():
    port = os.environ["PORT"]



app = Flask(__name__,static_url_path = "/front-end")

object = analyze()

@app.route('/',methods=["GET"])
def front_end():
    print("hit")
    print(app.send_static_file('/front-end/front-end.html'))
    return app.send_static_file('/front-end/front-end.html')

@app.route('/<text>',methods=["POST"])
def sentiment_analyze(text):
    result = object.input(text)
    print(text,result)
    return str(result)

app.run(host=host,port=port,debug=True)