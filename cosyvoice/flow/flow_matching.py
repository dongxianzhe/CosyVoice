import torch
from torch import Tensor
from matcha.models.components.flow_matching import BASECFM
from cosyvoice.utils.common import set_all_random_seed


class CausalConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(n_feats=in_channels, cfm_params=cfm_params, n_spks=n_spks, spk_emb_dim=spk_emb_dim)
        set_all_random_seed(0)
        self.rand_noise = torch.randn([1, 80, 50 * 300])
        self.t_scheduler = cfm_params.t_scheduler
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator

    @torch.inference_mode()
    def forward(self, mu: Tensor, mask: Tensor, n_timesteps: int, temperature: float=1.0, spks: Tensor | None=None, cond: Tensor | None=None, streaming: bool=False) -> Tensor:
        # mu (batch_size, hidden_size, mel_timesteps)
        # mask (batch_size, 1, mel_timesteps)
        # spks shape: (batch_size, hidden_size)
        # cond (batch_size, hidden_size, mel_timesteps)
        x: Tensor = self.rand_noise[:, :, :mu.size(2)].to(mu.device).to(mu.dtype) * temperature
        # fix prompt and overlap part mu and z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span: Tensor = 1 - torch.cos(t_span * 0.5 * torch.pi)

        # generated mel-spectrogram (batch_size, n_feats, mel_timesteps)
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=spks.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=spks.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=spks.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        for step in range(1, len(t_span)):
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond
            dphi_dt = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming)
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()