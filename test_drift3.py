import torch
from spiking_dreamcoder import SpikingDreamCoder, SpikeTemporalEncoder
import torch.nn as nn

def test():
    torch.manual_seed(42)
    encoder = SpikeTemporalEncoder(t_sym=6, channels=3)
    model = SpikingDreamCoder(channels_in=3, hidden_size=16)
    
    seq = '1 0 1'
    spikes_baseline = encoder.encode(seq)
    
    x_t_base = spikes_baseline[:, 0, :]
    out_win_base = model.resonator.w_in(x_t_base)
    
    # 1. Expand channels
    new_channel_idx = model.resonator.expand_input_channels(1)
    encoder.expand_channels(1)
    
    spikes_after = encoder.encode(seq)
    x_t_after = spikes_after[:, 0, :]
    out_win_after = model.resonator.w_in(x_t_after)
    
    diff_win = (out_win_after - out_win_base).abs().max().item()
    print(f"w_in diff: {diff_win}")
    
    out_norm_base = model.resonator.norm(out_win_base)
    out_norm_after = model.resonator.norm(out_win_after)
    diff_norm = (out_norm_after - out_norm_base).abs().max().item()
    print(f"norm diff: {diff_norm}")

if __name__ == "__main__":
    test()
