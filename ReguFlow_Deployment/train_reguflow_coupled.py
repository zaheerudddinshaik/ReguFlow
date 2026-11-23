import torch
import torch.nn as nn

import train_reguflow as base


class CoupledReguFlowModel(nn.Module):
    """
    Cross-coupled continuous and regulatory recurrent states.

    Both states interact at every historical time step.
    """

    def __init__(self):
        super().__init__()

        self.temporal_input = nn.Linear(
            base.TEMPORAL_FEATURES_PER_STEP,
            base.TEMPORAL_HIDDEN_SIZE,
        )

        self.regulatory_input = nn.Linear(
            base.REGULATORY_FEATURES_PER_STEP,
            base.REGULATORY_HIDDEN_SIZE,
        )

        self.continuous_update = nn.Linear(
            base.TEMPORAL_HIDDEN_SIZE * 2
            + base.REGULATORY_HIDDEN_SIZE,
            base.TEMPORAL_HIDDEN_SIZE,
        )

        self.regulatory_update = nn.Linear(
            base.REGULATORY_HIDDEN_SIZE * 2
            + base.TEMPORAL_HIDDEN_SIZE,
            base.REGULATORY_HIDDEN_SIZE,
        )

        self.temporal_projection = nn.Linear(
            base.TEMPORAL_HIDDEN_SIZE,
            base.FUSION_SIZE,
        )

        self.regulatory_projection = nn.Linear(
            base.REGULATORY_HIDDEN_SIZE,
            base.FUSION_SIZE,
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(
                base.TEMPORAL_HIDDEN_SIZE
                + base.REGULATORY_HIDDEN_SIZE,
                base.FUSION_SIZE,
            ),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(base.FUSION_SIZE, base.DECODER_SIZE),
            nn.ReLU(),
            nn.Linear(base.DECODER_SIZE, 6),
        )

    def forward(self, temporal_x, regulatory_x):

        batch_size, sequence_length, _ = temporal_x.shape

        z = temporal_x.new_zeros(
            batch_size,
            base.TEMPORAL_HIDDEN_SIZE,
        )

        r = temporal_x.new_zeros(
            batch_size,
            base.REGULATORY_HIDDEN_SIZE,
        )

        for t in range(sequence_length):

            temporal_candidate = torch.tanh(
                self.temporal_input(temporal_x[:, t, :])
            )

            regulatory_candidate = torch.tanh(
                self.regulatory_input(regulatory_x[:, t, :])
            )

            z = torch.tanh(
                self.continuous_update(
                    torch.cat(
                        [temporal_candidate, z, r],
                        dim=1,
                    )
                )
            )

            r = torch.tanh(
                self.regulatory_update(
                    torch.cat(
                        [regulatory_candidate, r, z],
                        dim=1,
                    )
                )
            )

        gate = self.fusion_gate(
            torch.cat([z, r], dim=1)
        )

        fused_state = (
            gate * self.temporal_projection(z)
            + (1.0 - gate) * self.regulatory_projection(r)
        )

        return self.decoder(fused_state)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--horizon",
        type=int,
        choices=base.HORIZONS,
        default=None,
    )
    args = parser.parse_args()

    # Replace the model used by the existing training functions.
    base.ReguFlowModel = CoupledReguFlowModel

    # Keep all original model checkpoints untouched.
    base.MODEL_DIR = (
        base.BASE_DIR / "reguflow_coupled_models"
    )

    base.validate_feature_configuration()

    if args.smoke_test:
        model = CoupledReguFlowModel().to(base.DEVICE)
        model.train()

        temporal = torch.randn(
            4, 48, base.TEMPORAL_FEATURES_PER_STEP,
            device=base.DEVICE,
        )
        regulatory = torch.randn(
            4, 48, base.REGULATORY_FEATURES_PER_STEP,
            device=base.DEVICE,
        )
        targets = torch.randn(4, 6, device=base.DEVICE)

        predictions = model(temporal, regulatory)

        assert predictions.shape == (4, 6)
        assert torch.isfinite(predictions).all()

        loss = nn.HuberLoss()(predictions, targets)
        loss.backward()

        for name in [
            "temporal_input",
            "regulatory_input",
            "continuous_update",
            "regulatory_update",
            "fusion_gate",
            "decoder",
        ]:
            module = getattr(model, name)
            gradients = [
                p.grad for p in module.parameters()
            ]
            assert all(
                g is not None and torch.isfinite(g).all()
                for g in gradients
            ), f"Gradient check failed: {name}"

        print("Forward pass: PASSED")
        print("Backward pass: PASSED")
        print("Cross-coupled gradient checks: PASSED")

        base.train_horizon(1, smoke_test=True)

        print("\nCOUPLED REGUFLOW SMOKE TEST PASSED.")
        return

    horizons = (
        [args.horizon]
        if args.horizon is not None
        else base.HORIZONS
    )

    for horizon in horizons:
        base.train_horizon(horizon)

    print("\nCOUPLED REGUFLOW TRAINING COMPLETED.")


if __name__ == "__main__":
    main()
