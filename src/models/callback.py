from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt


class PolicyGradientLossCallback(BaseCallback):
    """
    A custom callback class that logs the policy_gradient_loss during training.
    This class extends BaseCallback and used to capture and store the metrics we want.
    """

    def __init__(self, verbose=0):
        super(PolicyGradientLossCallback, self).__init__(verbose)
        self.losses = []

    def _on_step(self) -> bool:
        if hasattr(self.model, "logger"):
            logs = self.model.logger.name_to_value
            if "train/policy_gradient_loss" in logs:
                loss = logs["train/policy_gradient_loss"]
                self.losses.append(loss)
        return True

    def _on_training_end(self):
        """Plot the loss after training ends"""
        name = self.model.__class__.__name__
        plt.figure(figsize=(12, 4))
        plt.plot(self.losses, label="Policy Gradient Loss")
        plt.title(f"{name} - Policy Gradient Loss During Training")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
