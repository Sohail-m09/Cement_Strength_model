from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing
from src.model_build import model_build
from src.model_evaluation import model_evaluation
import pickle

def main():
    # 1. Ingest
    df = data_ingestion()
    
    # 2. Preprocess (Ensure your file has the 'return' statement!)
    X_train, X_test, y_train, y_test = data_preprocessing(df)
    
    # 3. Build (Pass df if your function requires it)
    models = model_build(df)
    
    # 4. Evaluate and get the best model back
    best_model, best_name = model_evaluation(models, X_train, y_train, X_test, y_test)

    # 5. Create the Pickle file
    pickle_filename = f"best_model_{best_name}.pkl"
    with open(pickle_filename, 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"Success! Saved {best_name} to {pickle_filename}")

if __name__ == "__main__":
    main()