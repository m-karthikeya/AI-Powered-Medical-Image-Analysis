#main
import os
import numpy as np
import cv2
from src.data_loader import get_data_generators
from src.model import build_model
from src.evaluate import plot_training_history
from src.config import DATA_DIR, EPOCHS, MODEL_SAVE_PATH, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, OUTPUTS_DIR

def create_dummy_data_if_needed(data_dir):
    """Creates dummy images if no real dataset is found to ensure the pipeline runs out-of-the-box."""
    train_dir = os.path.join(data_dir, "train")
    if not os.path.exists(os.path.join(train_dir, "NORMAL")):
        print("Real dataset not found. Generating dummy simulation data...")
        os.makedirs(os.path.join(train_dir, "NORMAL"), exist_ok=True)
        os.makedirs(os.path.join(train_dir, "PNEUMONIA"), exist_ok=True)
        
        for i in range(10):
            cv2.imwrite(os.path.join(train_dir, "NORMAL", f"img_{i}.jpg"), np.random.randint(0, 255, (IMG_HEIGHT, IMG_WIDTH, 3)))
            cv2.imwrite(os.path.join(train_dir, "PNEUMONIA", f"img_{i}.jpg"), np.random.randint(0, 255, (IMG_HEIGHT, IMG_WIDTH, 3)))

if __name__ == "__main__":
    print("=== AI Medical Image Analysis System ===")
    
    create_dummy_data_if_needed(DATA_DIR)

    train_gen, val_gen = get_data_generators(DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

    print("Building Model...")
    model = build_model(IMG_HEIGHT, IMG_WIDTH)
    model.summary()

    print("Starting Training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    print("Saving Model & Graphs...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    
    plot_training_history(history, OUTPUTS_DIR)
    
    print("=== Pipeline Complete! ===")