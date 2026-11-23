import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "reguflow_scaled"
MODEL_DIR = BASE_DIR / "reguflow_models"

HORIZONS = [1, 6, 12, 24]

POLLUTANTS = [
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO2 (µg/m³)",
    "SO2 (µg/m³)",
    "CO (mg/m³)",
    "Ozone (µg/m³)",
]

METEOROLOGY = [
    "AT (°C)",
    "RH (%)",
    "WS (m/s)",
    "WD (deg)",
    "RF (mm)",
    "TOT-RF (mm)",
    "SR (W/mt2)",
    "BP (mmHg)",
]

REGUFLOW_FLUX_FEATURES = [
    "PM2.5_Boundary_Velocity",
    "PM2.5_ReguFlow_Signed_Distance",
    "PM2.5_Boundary_Flux",
    "PM10_Boundary_Velocity",
    "PM10_ReguFlow_Signed_Distance",
    "PM10_Boundary_Flux",
    "NO2_Boundary_Velocity",
    "NO2_ReguFlow_Signed_Distance",
    "NO2_Boundary_Flux",
    "SO2_Boundary_Velocity",
    "SO2_ReguFlow_Signed_Distance",
    "SO2_Boundary_Flux",
    "CO_Boundary_Velocity",
    "CO_ReguFlow_Signed_Distance",
    "CO_Boundary_Flux",
    "Ozone_Boundary_Velocity",
    "Ozone_ReguFlow_Signed_Distance",
    "Ozone_Boundary_Flux",
    "ReguFlow_Top2_Dominance_Gap",
    "ReguFlow_Competition_Flux",
]

REGULATORY_NUMERIC = [
    "PM2.5 (µg/m³)__subindex",
    "PM10 (µg/m³)__subindex",
    "NO2 (µg/m³)__subindex",
    "SO2 (µg/m³)__subindex",
    "CO (mg/m³)__subindex",
    "Ozone (µg/m³)__subindex",
    "AQI",
    "Second_Subindex",
    "Dominance_Gap",
    "Valid_Pollutant_Count",
    "Valid_Subindex_Count",
    "AQI_Boundary_Distance",
    "Nearest_AQI_Boundary",
    "PM2.5_Boundary_Distance",
    "PM10_Boundary_Distance",
    "NO2_Boundary_Distance",
    "SO2_Boundary_Distance",
    "CO_Boundary_Distance",
    "Ozone_Boundary_Distance",
    "CoElevated_Count_Above_50",
    "CoElevated_Count_Above_100",
    "Dominant_PM2.5",
    "Dominant_PM10",
    "Dominant_NO2",
    "Dominant_SO2",
    "Dominant_CO",
    "Dominant_Ozone",
    "AQI_Boundary_Velocity",
    "Dominance_Stability_12h",
    *REGUFLOW_FLUX_FEATURES,
]

# Reconstruct the same groups used by the dataset preparation script.
MEASUREMENT_FEATURES = POLLUTANTS + METEOROLOGY

SOURCE_FLAG_FEATURES = [
    f"{name}__was_missing"
    for name in (
        POLLUTANTS
        + [
            name for name in METEOROLOGY
            if name != "TOT-RF (mm)"
        ]
    )
]

REGULATORY_FLAG_FEATURES = [
    f"{name}__is_missing"
    for name in REGULATORY_NUMERIC
]

TEMPORAL_FEATURES = (
    MEASUREMENT_FEATURES + SOURCE_FLAG_FEATURES
)

REGULATORY_FEATURES = (
    REGULATORY_NUMERIC + REGULATORY_FLAG_FEATURES
)

SEQUENCE_LENGTH = 48
TEMPORAL_FEATURES_PER_STEP = 27
REGULATORY_FEATURES_PER_STEP = 98
TOTAL_FEATURES_PER_STEP = 125

EXPECTED_INPUT_FEATURES = (
    SEQUENCE_LENGTH * TOTAL_FEATURES_PER_STEP
)

TEMPORAL_HIDDEN_SIZE = 64
REGULATORY_HIDDEN_SIZE = 64
FUSION_SIZE = 64
DECODER_SIZE = 64

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 10
PATIENCE = 3
SEED = 42

TARGET_COLUMNS = [
    f"Target_{pollutant}"
    for pollutant in POLLUTANTS
]

TIME_COLUMNS = [
    "Forecast_Issue_Time",
    "Target_Time",
]

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. REGUFLOW MODEL
# ============================================================

class ReguFlowModel(nn.Module):
    """
    Two-branch ReguFlow architecture.

    Branch 1:
        Historical pollutant and meteorological information.

    Branch 2:
        Regulatory geometry, boundary flux and
        competition-related information.

    Fusion:
        Learned sigmoid gate combines both encoded states.

    Decoder:
        Predicts six future pollutant concentrations.
    """

    def __init__(self):
        super().__init__()

        # Temporal encoder: 27 features per time step.
        self.temporal_encoder = nn.GRU(
            input_size=TEMPORAL_FEATURES_PER_STEP,
            hidden_size=TEMPORAL_HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )

        # Regulatory encoder: 98 features per time step.
        self.regulatory_encoder = nn.GRU(
            input_size=REGULATORY_FEATURES_PER_STEP,
            hidden_size=REGULATORY_HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )

        # Gate decides how much information to use
        # from each branch.
        self.fusion_gate = nn.Sequential(
            nn.Linear(
                TEMPORAL_HIDDEN_SIZE + REGULATORY_HIDDEN_SIZE,
                FUSION_SIZE,
            ),
            nn.Sigmoid(),
        )

        # Project regulatory state to fusion dimension.
        self.regulatory_projection = nn.Linear(
            REGULATORY_HIDDEN_SIZE,
            FUSION_SIZE,
        )

        # Project temporal state to fusion dimension.
        self.temporal_projection = nn.Linear(
            TEMPORAL_HIDDEN_SIZE,
            FUSION_SIZE,
        )

        # Six-pollutant prediction decoder.
        self.decoder = nn.Sequential(
            nn.Linear(FUSION_SIZE, DECODER_SIZE),
            nn.ReLU(),
            nn.Linear(DECODER_SIZE, 6),
        )

    def forward(self, temporal_x, regulatory_x):

        _, temporal_hidden = self.temporal_encoder(
            temporal_x
        )

        _, regulatory_hidden = self.regulatory_encoder(
            regulatory_x
        )

        temporal_state = temporal_hidden[-1]
        regulatory_state = regulatory_hidden[-1]

        combined_state = torch.cat(
            [temporal_state, regulatory_state],
            dim=1,
        )

        gate = self.fusion_gate(combined_state)

        temporal_projected = self.temporal_projection(
            temporal_state
        )

        regulatory_projected = self.regulatory_projection(
            regulatory_state
        )

        fused_state = (
            gate * temporal_projected
            + (1.0 - gate) * regulatory_projected
        )

        predictions = self.decoder(fused_state)

        return predictions


# ============================================================
# 3. DATASET PATHS
# ============================================================

def dataset_path(horizon, split):

    filename = (
        f"HB_Colony_Vijayawada_{horizon}h_reguflow_"
        f"{split}_scaled.csv"
    )

    return DATA_DIR / filename


# ============================================================
# 4. FEATURE VALIDATION
# ============================================================

def validate_feature_configuration():

    if len(REGULATORY_NUMERIC) != 49:
        raise ValueError(
            f"Expected 49 regulatory numeric features, "
            f"found {len(REGULATORY_NUMERIC)}."
        )

    if len(REGUFLOW_FLUX_FEATURES) != 20:
        raise ValueError(
            "Expected exactly 20 ReguFlow flux features."
        )

    if len(TEMPORAL_FEATURES) != TEMPORAL_FEATURES_PER_STEP:
        raise ValueError(
            f"Expected 27 temporal features, "
            f"found {len(TEMPORAL_FEATURES)}."
        )

    if len(REGULATORY_FEATURES) != REGULATORY_FEATURES_PER_STEP:
        raise ValueError(
            f"Expected 98 regulatory features, "
            f"found {len(REGULATORY_FEATURES)}."
        )

    if len(set(TEMPORAL_FEATURES)) != len(TEMPORAL_FEATURES):
        raise ValueError("Duplicate temporal feature names.")

    if len(set(REGULATORY_FEATURES)) != len(REGULATORY_FEATURES):
        raise ValueError("Duplicate regulatory feature names.")

    if set(TEMPORAL_FEATURES) & set(REGULATORY_FEATURES):
        raise ValueError(
            "Temporal and regulatory feature groups overlap."
        )

    print("\nFeature configuration verified:")
    print(f"  Temporal features: {len(TEMPORAL_FEATURES)}")
    print(f"  Regulatory features: {len(REGULATORY_FEATURES)}")
    print(f"  Total per time step: {TOTAL_FEATURES_PER_STEP}")
    print(f"  Sequence length: {SEQUENCE_LENGTH}")
    print(f"  Total lagged inputs: {EXPECTED_INPUT_FEATURES}")


# ============================================================
# 5. LOAD AND PREPARE DATA
# ============================================================

def load_split(horizon, split, max_rows=None):

    path = dataset_path(horizon, split)

    if not path.exists():
        raise FileNotFoundError(
            f"ReguFlow dataset not found:\n{path}"
        )

    print(f"\nLoading {split}: {path.name}")

    df = pd.read_csv(
        path,
        nrows=max_rows,
        low_memory=False,
    )

    # Validate target and timestamp columns.
    required_columns = TARGET_COLUMNS + TIME_COLUMNS

    missing_required = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_required:
        raise ValueError(
            f"Missing required columns in {path.name}: "
            f"{missing_required}"
        )

    # Validate feature columns for every lag.
    all_expected_features = []

    for lag in range(SEQUENCE_LENGTH):

        suffix = f"__lag_{lag:02d}"

        expected_at_lag = (
            TEMPORAL_FEATURES + REGULATORY_FEATURES
        )

        lag_columns = [
            f"{feature}{suffix}"
            for feature in expected_at_lag
        ]

        missing_lag = [
            col for col in lag_columns
            if col not in df.columns
        ]

        if missing_lag:
            raise ValueError(
                f"{path.name}: Missing features at lag "
                f"{lag:02d}: {missing_lag[:10]}"
            )

        if len(lag_columns) != TOTAL_FEATURES_PER_STEP:
            raise ValueError(
                f"Incorrect feature count at lag {lag:02d}."
            )

        all_expected_features.extend(lag_columns)

    if len(all_expected_features) != EXPECTED_INPUT_FEATURES:
        raise ValueError(
            "Unexpected total number of input features."
        )

    # Check duplicate column names.
    if df.columns.duplicated().any():
        raise ValueError(
            f"Duplicate columns found in {path.name}."
        )

    # Verify target values.
    if df[TARGET_COLUMNS].isna().any().any():
        raise ValueError(
            f"Missing target values in {path.name}."
        )

    # Build oldest-to-newest sequences.
    #
    # lag_47 = oldest
    # lag_00 = newest
    #
    # Feature order within each time step follows
    # the original dataset preparation script.

    temporal_blocks = []
    regulatory_blocks = []

    for lag in range(SEQUENCE_LENGTH - 1, -1, -1):

        suffix = f"__lag_{lag:02d}"

        temporal_columns = [
            f"{feature}{suffix}"
            for feature in TEMPORAL_FEATURES
        ]

        regulatory_columns = [
            f"{feature}{suffix}"
            for feature in REGULATORY_FEATURES
        ]

        temporal_blocks.append(
            df[temporal_columns].to_numpy(dtype=np.float32)
        )

        regulatory_blocks.append(
            df[regulatory_columns].to_numpy(dtype=np.float32)
        )

    temporal_x = np.stack(
        temporal_blocks,
        axis=1,
    )

    regulatory_x = np.stack(
        regulatory_blocks,
        axis=1,
    )

    y = df[TARGET_COLUMNS].to_numpy(
        dtype=np.float32
    )

    # Check all arrays for finite values.
    if not np.isfinite(temporal_x).all():
        raise ValueError(
            f"Non-finite temporal inputs in {path.name}."
        )

    if not np.isfinite(regulatory_x).all():
        raise ValueError(
            f"Non-finite regulatory inputs in {path.name}."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            f"Non-finite targets in {path.name}."
        )

    # Verify expected tensor shapes.
    if temporal_x.shape[1:] != (
        SEQUENCE_LENGTH,
        TEMPORAL_FEATURES_PER_STEP,
    ):
        raise ValueError(
            f"Incorrect temporal tensor shape: {temporal_x.shape}"
        )

    if regulatory_x.shape[1:] != (
        SEQUENCE_LENGTH,
        REGULATORY_FEATURES_PER_STEP,
    ):
        raise ValueError(
            f"Incorrect regulatory tensor shape: {regulatory_x.shape}"
        )

    if y.shape[1] != 6:
        raise ValueError(
            f"Expected six target pollutants, got {y.shape[1]}."
        )

    print(f"Rows loaded: {len(df):,}")
    print(f"Temporal input shape: {temporal_x.shape}")
    print(f"Regulatory input shape: {regulatory_x.shape}")
    print(f"Target shape: {y.shape}")

    return temporal_x, regulatory_x, y


# ============================================================
# 6. SMOKE TEST
# ============================================================

def run_smoke_test():

    print("\n" + "=" * 65)
    print("REGUFLOW MODEL SMOKE TEST")
    print("=" * 65)

    temporal_x, regulatory_x, y = load_split(
        horizon=1,
        split="train",
        max_rows=256,
    )

    if len(y) < 4:
        raise ValueError(
            "Not enough rows for ReguFlow smoke test."
        )

    model = ReguFlowModel().to(DEVICE)

    temporal_tensor = torch.from_numpy(
        temporal_x[:4]
    ).to(DEVICE)

    regulatory_tensor = torch.from_numpy(
        regulatory_x[:4]
    ).to(DEVICE)

    target_tensor = torch.from_numpy(
        y[:4]
    ).to(DEVICE)

    model.eval()

    with torch.no_grad():

        predictions = model(
            temporal_tensor,
            regulatory_tensor,
        )

    if predictions.shape != (4, 6):
        raise ValueError(
            f"Unexpected prediction shape: {predictions.shape}"
        )

    criterion = nn.HuberLoss()

    loss = criterion(
        predictions,
        target_tensor,
    )

    if not torch.isfinite(loss):
        raise ValueError(
            "Smoke-test loss is not finite."
        )

    print("\nSmoke-test results:")
    print("  Temporal input:", tuple(temporal_tensor.shape))
    print("  Regulatory input:", tuple(regulatory_tensor.shape))
    print("  Target:", tuple(target_tensor.shape))
    print("  Prediction:", tuple(predictions.shape))
    print(f"  Initial Huber loss: {loss.item():.6f}")

    # Also verify that gradients flow through both branches.
    model.train()

    model.zero_grad()

    predictions = model(
        temporal_tensor,
        regulatory_tensor,
    )

    loss = criterion(
        predictions,
        target_tensor,
    )

    loss.backward()

    temporal_gradient = (
        model.temporal_encoder.weight_ih_l0.grad
    )

    regulatory_gradient = (
        model.regulatory_encoder.weight_ih_l0.grad
    )

    if temporal_gradient is None:
        raise RuntimeError(
            "Temporal encoder did not receive gradients."
        )

    if regulatory_gradient is None:
        raise RuntimeError(
            "Regulatory encoder did not receive gradients."
        )

    print("  Temporal encoder gradient: verified")
    print("  Regulatory encoder gradient: verified")
    print("  Gated fusion forward/backward: verified")
    print("\nREGUFLOW SMOKE TEST PASSED.")


# ============================================================
# 7. TRAIN ONE FORECASTING HORIZON
# ============================================================

def train_horizon(horizon, smoke_test=False):

    print("\n" + "=" * 65)
    print(f"TRAINING REGUFLOW: {horizon}-HOUR FORECAST")
    print("=" * 65)

    train_limit = 256 if smoke_test else None
    validation_limit = 64 if smoke_test else None

    temporal_train, regulatory_train, y_train = load_split(
        horizon,
        "train",
        max_rows=train_limit,
    )

    temporal_val, regulatory_val, y_val = load_split(
        horizon,
        "validation",
        max_rows=validation_limit,
    )

    if smoke_test:
        temporal_train = temporal_train[:256]
        regulatory_train = regulatory_train[:256]
        y_train = y_train[:256]

        temporal_val = temporal_val[:64]
        regulatory_val = regulatory_val[:64]
        y_val = y_val[:64]

    if len(y_train) == 0 or len(y_val) == 0:
        raise ValueError(
            "Training or validation dataset is empty."
        )

    # Target standardization using training data only.
    target_mean = y_train.mean(axis=0)
    target_std = y_train.std(axis=0)

    target_std[target_std < 1e-8] = 1.0

    y_train_scaled = (
        (y_train - target_mean) / target_std
    ).astype(np.float32)

    y_val_scaled = (
        (y_val - target_mean) / target_std
    ).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(temporal_train),
        torch.from_numpy(regulatory_train),
        torch.from_numpy(y_train_scaled),
    )

    validation_dataset = TensorDataset(
        torch.from_numpy(temporal_val),
        torch.from_numpy(regulatory_val),
        torch.from_numpy(y_val_scaled),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = ReguFlowModel().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.HuberLoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    epochs_to_run = 1 if smoke_test else MAX_EPOCHS

    history = []

    for epoch in range(1, epochs_to_run + 1):

        # -------------------------
        # Training phase
        # -------------------------

        model.train()

        train_total = 0.0

        for temporal_batch, regulatory_batch, target_batch in train_loader:

            temporal_batch = temporal_batch.to(DEVICE)
            regulatory_batch = regulatory_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(
                temporal_batch,
                regulatory_batch,
            )

            loss = criterion(
                predictions,
                target_batch,
            )

            loss.backward()

            optimizer.step()

            train_total += loss.item() * len(target_batch)

        train_loss = (
            train_total / len(train_dataset)
        )

        # -------------------------
        # Validation phase
        # -------------------------

        model.eval()

        validation_total = 0.0

        with torch.no_grad():

            for temporal_batch, regulatory_batch, target_batch in validation_loader:

                temporal_batch = temporal_batch.to(DEVICE)
                regulatory_batch = regulatory_batch.to(DEVICE)
                target_batch = target_batch.to(DEVICE)

                predictions = model(
                    temporal_batch,
                    regulatory_batch,
                )

                loss = criterion(
                    predictions,
                    target_batch,
                )

                validation_total += (
                    loss.item() * len(target_batch)
                )

        validation_loss = (
            validation_total / len(validation_dataset)
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        })

        print(
            f"Epoch {epoch}/{epochs_to_run} | "
            f"Train loss: {train_loss:.6f} | "
            f"Validation loss: {validation_loss:.6f}"
        )

        # -------------------------
        # Best checkpoint tracking
        # -------------------------

        if validation_loss < best_val_loss:

            best_val_loss = validation_loss
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            not smoke_test
            and epochs_without_improvement >= PATIENCE
        ):
            print("Early stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError(
            "No valid ReguFlow checkpoint was produced."
        )

    if smoke_test:
        print("\nReguFlow one-epoch training smoke test passed.")
        return

    # Restore the best validation checkpoint.
    model.load_state_dict(best_state)

    output_dir = MODEL_DIR / f"{horizon}h"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "horizon_hours": horizon,
        "sequence_length": SEQUENCE_LENGTH,
        "temporal_features_per_step": TEMPORAL_FEATURES_PER_STEP,
        "regulatory_features_per_step": REGULATORY_FEATURES_PER_STEP,
        "temporal_hidden_size": TEMPORAL_HIDDEN_SIZE,
        "regulatory_hidden_size": REGULATORY_HIDDEN_SIZE,
        "fusion_size": FUSION_SIZE,
        "decoder_size": DECODER_SIZE,
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "target_columns": TARGET_COLUMNS,
        "temporal_features": TEMPORAL_FEATURES,
        "regulatory_features": REGULATORY_FEATURES,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
    }

    checkpoint_path = (
        output_dir / "reguflow_best.pt"
    )

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    metadata = {
        "model": "ReguFlow",
        "horizon_hours": horizon,
        "train_rows": len(y_train),
        "validation_rows": len(y_val),
        "sequence_length": SEQUENCE_LENGTH,
        "temporal_features_per_step": TEMPORAL_FEATURES_PER_STEP,
        "regulatory_features_per_step": REGULATORY_FEATURES_PER_STEP,
        "total_features_per_step": TOTAL_FEATURES_PER_STEP,
        "total_lagged_features": EXPECTED_INPUT_FEATURES,
        "temporal_hidden_size": TEMPORAL_HIDDEN_SIZE,
        "regulatory_hidden_size": REGULATORY_HIDDEN_SIZE,
        "fusion_size": FUSION_SIZE,
        "decoder_size": DECODER_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "maximum_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "loss": "HuberLoss",
        "optimizer": "Adam",
        "target_standardization": "Training data only",
        "target_columns": TARGET_COLUMNS,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "device": str(DEVICE),
        "training_history": history,
    }

    metadata_path = (
        output_dir / "training_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nBest model restored and saved.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metadata: {metadata_path}")


# ============================================================
# 8. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train the two-branch ReguFlow forecasting model."
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run input, architecture and one-epoch training checks.",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        choices=HORIZONS,
        default=None,
        help="Train only the selected forecasting horizon.",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("AQISTRATA - REGUFLOW TRAINING")
    print("=" * 65)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Model directory: {MODEL_DIR}")

    validate_feature_configuration()

    if args.smoke_test:

        if args.horizon not in (None, 1):
            raise ValueError(
                "The smoke test currently uses the 1-hour dataset. "
                "Run without --horizon or specify --horizon 1."
            )

        run_smoke_test()
        train_horizon(1, smoke_test=True)

        print("\nALL REGUFLOW SMOKE TESTS PASSED.")

    else:

        selected_horizons = (
            [args.horizon]
            if args.horizon is not None
            else HORIZONS
        )

        for horizon in selected_horizons:
            train_horizon(horizon)

        print("\nSelected ReguFlow training completed.")


if __name__ == "__main__":
    main()
