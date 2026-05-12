import subprocess
import os

models = ["xgb", "rf", "mlp", "cnn", "lstm", "cnnlstm", "rnn", "rnnlstm"]

def main():
    # Ensure we are in the correct directory or path is handled
    # Assuming this script is run from the project root or src/
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train.py")
    
    for model in models:
        print(f"\n==========================================")
        print(f"🚀 Training model: {model}")
        print(f"==========================================")
        
        try:
            # Run train.py with the model argument
            result = subprocess.run(
                ["python", train_script, "--model", model],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            print(f"✅ Finished training {model}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error training {model}:")
            print(e.stderr)

if __name__ == "__main__":
    main()
