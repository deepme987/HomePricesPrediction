from flask import Flask, request, jsonify, render_template
from server import util

app = Flask(__name__)
import os

@app.route('/')
def index():
    util.load_saved_artifacts()
    return render_template('app.html')


@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    city = request.args["city"]
    if city is None:
        city = ""

    response = jsonify({
        'locations': util.get_location_names(city)
    })

    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


@app.route('/predict_home_price', methods=['GET', 'POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    city = request.form['city']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price': util.get_estimated_price(city, location, total_sqft, bhk, bath)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    from server import util
    util.load_saved_artifacts()
    app.run(debug=True,threaded=False)
