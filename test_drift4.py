import torch
from spiking_dreamcoder import SpikingDreamCoder, SpikeTemporalEncoder
import torch.nn as nn

def test():
    torch.manual_seed(42)
    encoder = SpikeTemporalEncoder(t_sym=6, channels=3)
    model = SpikingDreamCoder(channels_in=3, hidden_size=16)
    
    seq = '1 0 1'
    spikes_baseline = encoder.encode(seq)
    
    with torch.no_grad():
        _, _, u_pre_baseline = model(spikes_baseline)
        
    x_t_base = spikes_baseline[:, 0, :]
    out_win_base = model.resonator.w_in(x_t_base)
    out_norm_base = model.resonator.norm(out_win_base)
    
    # 1. Expand channels
    new_channel_idx = model.resonator.expand_input_channels(1)
    model.resonator.add_lora_adapter(new_channel_idx)
    encoder.expand_channels(1)
    
    spikes_after = encoder.encode(seq)
    
    with torch.no_grad():
        _, _, u_pre_after = model(spikes_after)
        
    x_t_after = spikes_after[:, 0, :]
    out_win_after = model.resonator.w_in(x_t_after)
    out_norm_after = model.resonator.norm(out_win_after)
    
    print(f"u_pre_baseline t=0: {u_pre_baseline[:, 0, :]}")
    print(f"out_norm_base: {out_norm_base}")
    print(f"u_pre_after t=0: {u_pre_after[:, 0, :]}")
    print(f"out_norm_after: {out_norm_after}")

if __name__ == "__main__":
    test()
