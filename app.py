from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging
from flask_cors import CORS

app = Flask(__name__)


CORS(app)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('randomForest.pkl')
scaler = joblib.load('scalerR.pkl')
encoder_tipo = joblib.load('encoder_tipo.pkl')
encoder_categoria = joblib.load('encoder_categoria.pkl')
encoder_genero = joblib.load('encoder_genero.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        descuento = data.get('descuento')
        precio = data.get('precio')
        categoria = data.get('categoria')
        tipo = data.get('tipo')
        rating = data.get('rating')
        calificacion = data.get('calificacion')
        cantidad = data.get('cantidad')
        total = data.get('total')
        genero = data.get('genero')
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[descuento, precio, categoria, tipo, rating, calificacion, cantidad, total, genero]], 
                               columns=['descuento', 'precio', 'categoria', 'tipo', 'rating', 'calificacion', 'cantidad', 'total', 'genero'])
        app.logger.debug(f'DataFrame sin codificar: {data_df}')
        data_df['categoria'] = encoder_categoria.transform(data_df[['categoria']])
        data_df['tipo'] = encoder_tipo.transform(data_df[['tipo']])
        data_df['genero'] = encoder_genero.transform(data_df[['genero']])
        
        columns = data_df.columns
        data_df = scaler.transform(data_df)
        data_df = pd.DataFrame(data_df,columns=columns)
        
        app.logger.debug(f'DataFrame codificado: {data_df}')

        
        # Realizar predicciones 
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Convertir la predicción a un tipo de dato adecuado para la respuesta
        predicted_value = prediction[0]  # Asumimos que prediction[0] es un array de un solo elemento
        
        # Devolver las predicciones como respuesta JSON 
        print(f"la predicciones {predicted_value}")
        return jsonify({'tipo_cliente': float(predicted_value)})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)