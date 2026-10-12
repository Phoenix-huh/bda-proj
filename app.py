from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import datetime
import os
import logging
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model
try:
    model = joblib.load('models/air_quality_model.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Load stations for dropdown and validation
try:
    stations = pd.read_csv('data/stations.csv')
    # Create a list of dictionaries for dropdown
    station_list = [
        {'StationId': row['StationId'], 'display': f"{row['City']}, {row['State']}"}
        for _, row in stations[['StationId', 'City', 'State']].dropna().iterrows()
    ]
    logger.info(f"Loaded {len(station_list)} stations: {[s['StationId'] for s in station_list[:10]]}...")
except Exception as e:
    logger.error(f"Error loading stations.csv: {str(e)}")
    raise

# Load mean values for lag features
try:
    station_hour = pd.read_csv('data/station_hour_cleaned.csv', low_memory=False)
    targets = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 
               'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
    lag_means = {f'{target}_lag1': station_hour[target].mean() for target in targets}
    logger.info("Loaded mean values for lag features")
except Exception as e:
    logger.error(f"Error loading lag feature means: {str(e)}")
    lag_means = {}

# Load data for visualization (once at startup)
try:
    station_hour['Datetime'] = pd.to_datetime(station_hour['Datetime'])
    station_hour['Date'] = station_hour['Datetime'].dt.date  # For daily aggregation
    logger.info("Data loaded for visualization")
except Exception as e:
    logger.error(f"Error loading data for visualization: {str(e)}")
    station_hour = pd.DataFrame()  # Empty if error

@app.route('/')
def home():
    logger.debug("Rendering home page")
    return render_template('index.html', stations=station_list, targets=targets)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        station = request.form['station']
        date_str = request.form['date']
        hour = request.form['hour']
        logger.debug(f"Received form data: Station={station}, Date={date_str}, Hour={hour}")

        # Validate StationId
        if station not in [s['StationId'] for s in station_list]:
            logger.error(f"Invalid StationId: {station}")
            return jsonify({'error': f"Station {station} not found in stations.csv"})

        try:
            hour = int(hour)
            if not 0 <= hour <= 23:
                raise ValueError
        except ValueError:
            logger.error(f"Invalid hour: {hour}")
            return jsonify({'error': 'Hour must be an integer between 0 and 23'})

        try:
            date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            logger.debug(f"Parsed date: {date}")
        except ValueError:
            logger.error(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")
            return jsonify({'error': 'Date must be in YYYY-MM-DD format (e.g., 2017-01-01)'})

        input_df = pd.DataFrame({
            'StationId': [station],
            'Year': [date.year],
            'Month': [date.month],
            'Day': [date.day],
            'Hour': [hour],
            **{key: [value] for key, value in lag_means.items()}  # Add lag features with mean values
        })
        logger.debug(f"Input DataFrame:\n{input_df.to_string()}")

        prediction = model.predict(input_df)[0]
        logger.info(f"Prediction generated: {prediction}")

        pred_dict = dict(zip(targets, prediction))
        aqi = pred_dict['AQI']
        if aqi <= 50:
            bucket = 'Good'
        elif aqi <= 100:
            bucket = 'Satisfactory'
        elif aqi <= 200:
            bucket = 'Moderate'
        elif aqi <= 300:
            bucket = 'Poor'
        elif aqi <= 400:
            bucket = 'Very Poor'
        else:
            bucket = 'Severe'
        pred_dict['AQI_Bucket'] = bucket

        logger.info(f"Prediction result: {pred_dict}")
        return jsonify(pred_dict)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f"Prediction failed: {str(e)}"})

@app.route('/visualize', methods=['POST'])
def visualize():
    try:
        station = request.form['station']
        parameters = request.form.getlist('parameters')
        if not parameters:
            parameters = targets  # Default to all if none selected
        logger.debug(f"Visualization request: Station={station}, Parameters={parameters}")

        if station not in [s['StationId'] for s in station_list]:
            logger.error(f"Invalid StationId: {station}")
            return jsonify({'error': f"Station {station} not found"})

        # Filter data for station
        df_station = station_hour[station_hour['StationId'] == station]
        if df_station.empty:
            logger.error(f"No data for station: {station}")
            return jsonify({'error': f"No data available for station {station}"})

        # Compute averages for selected parameters
        averages = df_station[parameters].mean()

        # Generate Pie Diagram (distribution of averages)
        fig_pie, ax_pie = plt.subplots()
        averages.plot(kind='pie', autopct='%1.1f%%', ax=ax_pie, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#a3a3a3','#ff66b3','#99ffcc','#cc99ff','#ffcc66','#66ff99','#ccff99','#99ccff','#ff99cc'])
        ax_pie.set_title('Distribution of Selected Gases')
        ax_pie.set_ylabel('')
        buffer_pie = BytesIO()
        fig_pie.savefig(buffer_pie, format='png')
        buffer_pie.seek(0)
        pie_url = base64.b64encode(buffer_pie.getvalue()).decode('utf-8')

        # Generate Bar Graph (averages)
        fig_bar, ax_bar = plt.subplots()
        averages.plot(kind='bar', ax=ax_bar, color='#66b3ff')
        ax_bar.set_title('Average Values of Selected Gases')
        ax_bar.set_ylabel('Average Value')
        ax_bar.set_xlabel('Parameter')
        buffer_bar = BytesIO()
        fig_bar.savefig(buffer_bar, format='png')
        buffer_bar.seek(0)
        bar_url = base64.b64encode(buffer_bar.getvalue()).decode('utf-8')

        # Generate Line Chart (daily trend)
        daily_means = df_station.groupby('Date')[parameters].mean()
        fig_line, ax_line = plt.subplots()
        daily_means.plot(kind='line', ax=ax_line)
        ax_line.set_title('Daily Trend of Selected Gases')
        ax_line.set_ylabel('Value')
        ax_line.set_xlabel('Date')
        ax_line.legend(loc='upper right')
        buffer_line = BytesIO()
        fig_line.savefig(buffer_line, format='png')
        buffer_line.seek(0)
        line_url = base64.b64encode(buffer_line.getvalue()).decode('utf-8')

        logger.info("Visualizations generated successfully")
        return jsonify({'pie_url': pie_url, 'bar_url': bar_url, 'line_url': line_url})
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        return jsonify({'error': f"Visualization failed: {str(e)}"})

if __name__ == '__main__':
    logger.info("Starting Flask app")
    app.run(debug=True)
























# from flask import Flask, render_template, request, jsonify
# import joblib
# import pandas as pd
# import datetime
# import os
# import logging
# import json
# import matplotlib.pyplot as plt
# from io import BytesIO
# import base64

# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('app.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# app = Flask(__name__)

# # Load the model
# try:
#     model = joblib.load('models/air_quality_model_svm.pkl')  # Changed to SVM model
#     logger.info("Model loaded successfully")
# except Exception as e:
#     logger.error(f"Error loading model: {str(e)}")
#     raise

# # Load stations for dropdown and validation
# try:
#     stations = pd.read_csv('data/stations.csv')
#     station_list = [
#         {'StationId': row['StationId'], 'display': f"{row['City']}, {row['State']}"}
#         for _, row in stations[['StationId', 'City', 'State']].dropna().iterrows()
#     ]
#     logger.info(f"Loaded {len(station_list)} stations: {[s['StationId'] for s in station_list[:10]]}...")
# except Exception as e:
#     logger.error(f"Error loading stations.csv: {str(e)}")
#     raise

# # Load mean values for lag features
# try:
#     station_hour = pd.read_csv('data/station_hour_cleaned.csv', low_memory=False)
#     targets = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 
#                'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
#     lag_means = {f'{target}_lag1': station_hour[target].mean() for target in targets}
#     logger.info("Loaded mean values for lag features")
# except Exception as e:
#     logger.error(f"Error loading lag feature means: {str(e)}")
#     lag_means = {}

# # Load data for visualization (once at startup)
# try:
#     station_hour['Datetime'] = pd.to_datetime(station_hour['Datetime'])
#     station_hour['Date'] = station_hour['Datetime'].dt.date
#     logger.info("Data loaded for visualization")
# except Exception as e:
#     logger.error(f"Error loading data for visualization: {str(e)}")
#     station_hour = pd.DataFrame()

# @app.route('/')
# def home():
#     logger.debug("Rendering home page")
#     return render_template('index.html', stations=station_list, targets=targets)

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         station = request.form['station']
#         date_str = request.form['date']
#         hour = request.form['hour']
#         logger.debug(f"Received form data: Station={station}, Date={date_str}, Hour={hour}")

#         if station not in [s['StationId'] for s in station_list]:
#             logger.error(f"Invalid StationId: {station}")
#             return jsonify({'error': f"Station {station} not found in stations.csv"})

#         try:
#             hour = int(hour)
#             if not 0 <= hour <= 23:
#                 raise ValueError
#         except ValueError:
#             logger.error(f"Invalid hour: {hour}")
#             return jsonify({'error': 'Hour must be an integer between 0 and 23'})

#         try:
#             date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
#             logger.debug(f"Parsed date: {date}")
#         except ValueError:
#             logger.error(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")
#             return jsonify({'error': 'Date must be in YYYY-MM-DD format (e.g., 2017-01-01)'})

#         input_df = pd.DataFrame({
#             'StationId': [station],
#             'Year': [date.year],
#             'Month': [date.month],
#             'Day': [date.day],
#             'Hour': [hour],
#             **{key: [value] for key, value in lag_means.items()}
#         })
#         logger.debug(f"Input DataFrame:\n{input_df.to_string()}")

#         prediction = model.predict(input_df)[0]
#         logger.info(f"Prediction generated: {prediction}")

#         pred_dict = dict(zip(targets, prediction))
#         aqi = pred_dict['AQI']
#         if aqi <= 50:
#             bucket = 'Good'
#         elif aqi <= 100:
#             bucket = 'Satisfactory'
#         elif aqi <= 200:
#             bucket = 'Moderate'
#         elif aqi <= 300:
#             bucket = 'Poor'
#         elif aqi <= 400:
#             bucket = 'Very Poor'
#         else:
#             bucket = 'Severe'
#         pred_dict['AQI_Bucket'] = bucket

#         logger.info(f"Prediction result: {pred_dict}")
#         return jsonify(pred_dict)
#     except Exception as e:
#         logger.error(f"Prediction error: {str(e)}")
#         return jsonify({'error': f"Prediction failed: {str(e)}"})

# @app.route('/visualize', methods=['POST'])
# def visualize():
#     try:
#         station = request.form['station']
#         parameters = request.form.getlist('parameters')
#         if not parameters:
#             parameters = targets
#         logger.debug(f"Visualization request: Station={station}, Parameters={parameters}")

#         if station not in [s['StationId'] for s in station_list]:
#             logger.error(f"Invalid StationId: {station}")
#             return jsonify({'error': f"Station {station} not found"})

#         df_station = station_hour[station_hour['StationId'] == station]
#         if df_station.empty:
#             logger.error(f"No data for station: {station}")
#             return jsonify({'error': f"No data available for station {station}"})

#         averages = df_station[parameters].mean()

#         fig_pie, ax_pie = plt.subplots()
#         averages.plot(kind='pie', autopct='%1.1f%%', ax=ax_pie, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#a3a3a3','#ff66b3','#99ffcc','#cc99ff','#ffcc66','#66ff99','#ccff99','#99ccff','#ff99cc'])
#         ax_pie.set_title('Distribution of Selected Gases')
#         ax_pie.set_ylabel('')
#         buffer_pie = BytesIO()
#         fig_pie.savefig(buffer_pie, format='png')
#         buffer_pie.seek(0)
#         pie_url = base64.b64encode(buffer_pie.getvalue()).decode('utf-8')

#         fig_bar, ax_bar = plt.subplots()
#         averages.plot(kind='bar', ax=ax_bar, color='#66b3ff')
#         ax_bar.set_title('Average Values of Selected Gases')
#         ax_bar.set_ylabel('Average Value')
#         ax_bar.set_xlabel('Parameter')
#         buffer_bar = BytesIO()
#         fig_bar.savefig(buffer_bar, format='png')
#         buffer_bar.seek(0)
#         bar_url = base64.b64encode(buffer_bar.getvalue()).decode('utf-8')

#         daily_means = df_station.groupby('Date')[parameters].mean()
#         fig_line, ax_line = plt.subplots()
#         daily_means.plot(kind='line', ax=ax_line)
#         ax_line.set_title('Daily Trend of Selected Gases')
#         ax_line.set_ylabel('Value')
#         ax_line.set_xlabel('Date')
#         ax_line.legend(loc='upper right')
#         buffer_line = BytesIO()
#         fig_line.savefig(buffer_line, format='png')
#         buffer_line.seek(0)
#         line_url = base64.b64encode(buffer_line.getvalue()).decode('utf-8')

#         logger.info("Visualizations generated successfully")
#         return jsonify({'pie_url': pie_url, 'bar_url': bar_url, 'line_url': line_url})
#     except Exception as e:
#         logger.error(f"Visualization error: {str(e)}")
#         return jsonify({'error': f"Visualization failed: {str(e)}"})

# if __name__ == '__main__':
#     logger.info("Starting Flask app")
#     app.run(debug=True)