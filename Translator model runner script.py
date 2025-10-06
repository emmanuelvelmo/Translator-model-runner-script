import os # Módulo os para acceder a funciones del sistema operativo
import pathlib # Manejo de rutas de archivos de manera moderna
import numpy # Operaciones con arrays numéricos
import tensorflow # Librería para usar modelos de aprendizaje automático
import transformers # Biblioteca para modelos de lenguaje y procesamiento de texto

transformers.utils.logging.set_verbosity_error()  # Silenciar mensaje de advertencia

# FUNCIONES
# Función para generar traducciones del modelo
def f_traducir(modelo_val, texto_entrada, tokenizer_val):
    # Preprocesar el texto de entrada
    texto_procesado = tokenizer_val(texto_entrada, 
                                   return_tensors = "tf", 
                                   truncation = True)
    
    # Realizar la traducción
    traduccion_val = modelo_val.generate(
        input_ids = texto_procesado["input_ids"],
        attention_mask = texto_procesado["attention_mask"],
        num_beams = 4,
        early_stopping = True
    )
    
    # Decodificar la traducción a texto
    texto_traducido = tokenizer_val.decode(
        traduccion_val[0], 
        skip_special_tokens = True
    )
    
    return texto_traducido

# Función para cargar el modelo de traducción desde archivo
def cargar_modelo():
    # Directorio de carpeta de modelos
    carpeta_modelo = pathlib.Path("h5")
    
    # Crear carpeta si no existe
    if not carpeta_modelo.exists():
        carpeta_modelo.mkdir(exist_ok = True) # Crear carpeta si no existe
        
        input('Place model files in "h5" folder and press Enter...')
    
    # Cargar el modelo desde la carpeta
    modelo_val = transformers.TFAutoModelForSeq2SeqLM.from_pretrained(str(carpeta_modelo))
    
    # Cargar el tokenizador desde la carpeta
    tokenizer_val = transformers.AutoTokenizer.from_pretrained(str(carpeta_modelo))
    
    return modelo_val, tokenizer_val

# PUNTO DE PARTIDA
try:
    # Cargar el modelo y tokenizer al iniciar el programa
    modelo_traduccion, tokenizer_traduccion = cargar_modelo()

    print() # Salto de línea

    # Bucle principal del programa
    while True:
        texto_entrada = input("Input: ").strip()
        
        # Verificar si la entrada está vacía
        if not texto_entrada:
            print() # Salto de línea
            
            continue # Retornar al inicio del while

        print() # Salto de línea
        
        # Generar traducción
        texto_traducido = f_traducir(modelo_traduccion, texto_entrada, tokenizer_traduccion)
        
        print(texto_traducido + "\n") # Mostrar traducción con salto de línea al final

except Exception as e:
    print(f"Error loading model files: {str(e)}")
    
    # Detener el programa
    input()