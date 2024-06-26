
from flask import Flask, request, render_template
from src.crop_recommendation.pipelines.prediction_pipeline import PredictPipeline, CustomData

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    
    else:
        data = CustomData(
            N = int(request.form.get('N')),
            P = int(request.form.get('P')),
            K = int(request.form.get('K')),
            temperature = float(request.form.get('temperature')),
            humidity = float(request.form.get('humidity')),
            ph = float(request.form.get('ph')),
            rainfall = float(request.form.get('rainfall'))
        )
        
        print(data)
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        return render_template('result.html', results=results[0])

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
