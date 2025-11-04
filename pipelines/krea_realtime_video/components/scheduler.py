import types

import torch


class KreaSchedulerWrapper(torch.nn.Module):
    """
    A torch module wrapper for schedulers that sets timesteps and provides
    conversion methods (convert_x0_to_noise, convert_noise_to_x0, convert_velocity_to_x0).
    """

    def __init__(
        self, scheduler, num_inference_steps: int = 1000, training: bool = True
    ):
        """
        Initialize the wrapper.

        Args:
            scheduler: The scheduler instance to wrap
            num_inference_steps: Number of inference steps for set_timesteps
            training: Whether to set timesteps in training mode
        """
        super().__init__()
        self.scheduler = scheduler
        self.set_timesteps(num_inference_steps, training)

    def set_timesteps(self, num_inference_steps: int = 1000, training: bool = True):
        """
        Set timesteps on the scheduler and bind conversion methods to scheduler.

        Args:
            num_inference_steps: Number of inference steps
            training: Whether to set timesteps in training mode
        """
        self.scheduler.set_timesteps(num_inference_steps, training=training)

        # Bind conversion methods to scheduler
        self.scheduler.convert_x0_to_noise = types.MethodType(
            self.convert_x0_to_noise, self.scheduler
        )
        self.scheduler.convert_noise_to_x0 = types.MethodType(
            self.convert_noise_to_x0, self.scheduler
        )
        self.scheduler.convert_velocity_to_x0 = types.MethodType(
            self.convert_velocity_to_x0, self.scheduler
        )

    @staticmethod
    def _get_alphas_cumprod(scheduler_instance):
        """
        Get alphas_cumprod from a scheduler instance.
        If the scheduler doesn't have alphas_cumprod, compute it from sigmas.
        For FlowMatchScheduler: sigma corresponds to noise level, so alpha = 1 - sigma.
        """
        if hasattr(scheduler_instance, "alphas_cumprod"):
            return scheduler_instance.alphas_cumprod

        # For FlowMatchScheduler, compute alphas_cumprod from sigmas
        if hasattr(scheduler_instance, "sigmas"):
            # In flow matching: x_t = (1 - sigma_t) * x_0 + sigma_t * noise
            # So alpha_t = 1 - sigma_t, and alphas_cumprod = 1 - sigmas
            return 1 - scheduler_instance.sigmas

        raise AttributeError(
            "Scheduler must have either 'alphas_cumprod' or 'sigmas' attribute"
        )

    def convert_x0_to_noise(
        self, x0: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert the diffusion network's x0 prediction to noise prediction.
        x0: the predicted clean data with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        noise = (xt-sqrt(alpha_t)*x0) / sqrt(beta_t) (eq 11 in https://arxiv.org/abs/2311.18828)

        Note: When bound to scheduler via MethodType, self refers to the scheduler instance.
        """
        # use higher precision for calculations
        original_dtype = x0.dtype
        alphas_cumprod = KreaSchedulerWrapper._get_alphas_cumprod(self)
        x0, xt, alphas_cumprod = (
            x.double().to(x0.device) for x in [x0, xt, alphas_cumprod]
        )

        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        noise_pred = (xt - alpha_prod_t ** (0.5) * x0) / beta_prod_t ** (0.5)
        return noise_pred.to(original_dtype)

    def convert_noise_to_x0(
        self, noise: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert the diffusion network's noise prediction to x0 prediction.
        noise: the predicted noise with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        x0 = (x_t - sqrt(beta_t) * noise) / sqrt(alpha_t) (eq 11 in https://arxiv.org/abs/2311.18828)

        Note: When bound to scheduler via MethodType, self refers to the scheduler instance.
        """
        # use higher precision for calculations
        original_dtype = noise.dtype
        alphas_cumprod = KreaSchedulerWrapper._get_alphas_cumprod(self)
        noise, xt, alphas_cumprod = (
            x.double().to(noise.device) for x in [noise, xt, alphas_cumprod]
        )
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        x0_pred = (xt - beta_prod_t ** (0.5) * noise) / alpha_prod_t ** (0.5)
        return x0_pred.to(original_dtype)

    def convert_velocity_to_x0(
        self, velocity: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert the diffusion network's velocity prediction to x0 prediction.
        velocity: the predicted velocity with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        v = sqrt(alpha_t) * noise - sqrt(beta_t) x0
        noise = (xt-sqrt(alpha_t)*x0) / sqrt(beta_t)
        given v, x_t, we have
        x0 = sqrt(alpha_t) * x_t - sqrt(beta_t) * v
        see derivations https://chatgpt.com/share/679fb6c8-3a30-8008-9b0e-d1ae892dac56

        Note: When bound to scheduler via MethodType, self refers to the scheduler instance.
        """
        # use higher precision for calculations
        original_dtype = velocity.dtype
        alphas_cumprod = KreaSchedulerWrapper._get_alphas_cumprod(self)
        velocity, xt, alphas_cumprod = (
            x.double().to(velocity.device) for x in [velocity, xt, alphas_cumprod]
        )
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        x0_pred = (alpha_prod_t**0.5) * xt - (beta_prod_t**0.5) * velocity
        return x0_pred.to(original_dtype)

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped scheduler."""
        return getattr(self.scheduler, name)
