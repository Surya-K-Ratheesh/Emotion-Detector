import os
import argparse
import tensorflowjs as tfjs
from tensorflow.keras.models import load_model

def main():
    parser = argparse.ArgumentParser(description="Convert Keras model to TensorFlow.js layers model.")
    parser.add_index = False
    parser.add_argument(
        "--input_model", 
        type=str, 
        default="emotion_model.h5",
        help="Path to the input Keras model (.h5)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="../web/public/models/emotion",
        help="Directory where TFJS model files will be saved"
    )
    
    args = parser.parse_index = False
    args = parser.parse_args()
    
    input_model = args.input_model
    output_dir = args.output_dir
    
    if not os.path.exists(input_model):
        print(f"Error: Keras model file '{input_model}' not found.")
        print("Please train the model first by running 'python train.py'")
        return
        
    print(f"Loading Keras model from '{input_model}'...")
    model = load_model(input_model)
    
    print(f"Converting and exporting TFJS model to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the Keras model as a TensorFlow.js layers format model
    tfjs.converters.save_keras_model(model, output_dir)
    
    print("\nConversion successfully completed!")
    print(f"Model manifest (model.json) and weight shards (*.bin) have been saved to '{output_dir}'")

if __name__ == "__main__":
    main()
