"""Hyperparameter optimization for CNN using Optuna."""

import numpy as np, random, json
import sys
from pathlib import Path

# Add experiments directory to path
experiments_dir = Path(__file__).parent.parent
sys.path.insert(0, str(experiments_dir))

import optuna, tensorflow as tf

from sklearn.model_selection import train_test_split

from experiments.common.cnn_mlp import compile_mlp
from common.data_preprocessing import prepare_data_vectors

SEED = 42
M = 20
N_JOBS = 10 # if this is set too high, may run into database locks
N_TRIALS = 220

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(SEED)


def create_mlp_model(trial, input_shape):
    """Create a CNN model with hyperparameters suggested by Optuna trial."""
    # Suggest hyperparameters
    learning_rate = trial.suggest_categorical(
        "learning_rate", [0.01, 0.005, 0.001, 0.0005, 0.0001]
    )
    n_nodes = trial.suggest_categorical("n_nodes", [4, 8, 16, 32, 64, 128, 256])
    n_layers = trial.suggest_int("n_layers", 1, 2)
    dropout_rate = trial.suggest_categorical("dropout_rate", [0,0.25,0.5])

    model = compile_mlp(
        input_shape=input_shape,
        learning_rate=learning_rate,
        n_nodes=n_nodes,
        n_layers=n_layers,
        dropout_rate=dropout_rate
    )

    return model


def objective(trial, X_train, y_train, X_val, y_val, X_test, y_test, study_name):
    """Objective function for Optuna to minimize."""

    # Set random seeds for reproducibility
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Get input shape
    input_shape = X_train.shape[1:]

    # Early stopping callback
    patience = 10
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )

    # Create the model
    model = create_mlp_model(trial, input_shape)

    # Train the model on the fold
    model.fit(
        X_train,
        y_train,
        epochs=150,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0,
    )

    # Save architecture + weights
    model_path = (
        f"results/optuna/mlp/models/{study_name}_model_trial{trial.number}.keras"
    )
    model.save(model_path)
    trial.set_user_attr("model_path", model_path)

    # Save model history
    history_path = (
        f"results/optuna/mlp/history/{study_name}_history_trial{trial.number}.json"
    )
    history = model.history.history
    with open(history_path, 'w') as f:
        json.dump(history, f)
    trial.set_user_attr("history_path", history_path)

    # Evaluate the model on the train data
    loss_train, r2_score_train = model.evaluate(X_train, y_train, verbose=0)
    trial.set_user_attr("train_r2_score", r2_score_train)
    trial.set_user_attr("train_loss", loss_train)

    # Evaluate the model on the validation data
    loss_val, r2_score_val = model.evaluate(X_val, y_val, verbose=0)
    trial.set_user_attr("val_r2_score", r2_score_val)
    trial.set_user_attr("val_loss", loss_val)

    # Evaluate the model on the test set
    loss_test, r2_score_test = model.evaluate(X_test, y_test, verbose=0)
    trial.set_user_attr("test_r2_score", r2_score_test)
    trial.set_user_attr("test_loss", loss_test)

    return loss_val


if __name__ == "__main__":

    # Set up Optuna storage (SQLite database)
    storage_path = f"sqlite:///results/optuna/mlp/251121_mlp_gridsearch_M{M}_optuna.db"

    for N in [10, 5]:

        ################
        # Prepare data #
        ################

        (
            y_train_val,
            y_test,
            feature_names,
            feature_names_to_train,
            feature_names_to_test,
        ) = prepare_data_vectors(N, M)

        ##############
        # Run Optuna #
        ##############

        # perform hyperparameter optimization for each feature set
        for feature_name in feature_names:
            # Get training and validation data
            X_train_val = feature_names_to_train[feature_name]
            X_test = feature_names_to_test[feature_name]

            # Add channel dimension if missing
            if len(X_train_val.shape) == 3:
                X_train_val = X_train_val[..., np.newaxis]
                X_test = X_test[..., np.newaxis]

            # Split training data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val,
                y_train_val,
                test_size=0.2,
                random_state=SEED,
                stratify=y_train_val,
            )

            # Save the study results using Optuna's built-in storage
            study_name = f"{feature_name}_N{N}_M{M}"
            search_space = {
                "learning_rate": [0.01, 0.005, 0.001, 0.0005, 0.0001],
                "n_nodes": [4, 8, 16, 32, 64, 128, 256],
                "n_layers": [1,2],
                "dropout_rate": [0, 0.25, 0.5],
            }
            # Use GridSampler for exhaustive search over the defined grid
            grid_sampler = optuna.samplers.GridSampler(
                    seed=SEED, search_space=search_space
                )
            study = optuna.create_study(
                direction="minimize",
                storage=storage_path,
                load_if_exists=True,
                study_name=study_name,
                sampler=grid_sampler,
            )

            if grid_sampler.is_exhausted(study) == True:
                print(f"{study_name} is already finished.")
                continue
            else:
                study.optimize(
                    lambda trial: objective(
                        trial, X_train, y_train, X_val, y_val, X_test, y_test, study_name
                    ),
                    n_trials=N_TRIALS,
                    n_jobs=N_JOBS,
                    show_progress_bar=True,
                )
