import torch
from spiking_dreamcoder import SpikingDreamCoder, SpikeTemporalEncoder

def test():
    torch.manual_seed(42)
    encoder = SpikeTemporalEncoder(t_sym=6, channels=3)
    model = SpikingDreamCoder(channels_in=3, hidden_size=16)
    
    seq = '1 0 1'
    spikes_baseline = encoder.encode(seq)
    
    with torch.no_grad():
        _, _, u_pre_baseline = model(spikes_baseline)
        
    # 1. Expand channels
    new_channel_idx = model.resonator.expand_input_channels(1)
    model.resonator.add_lora_adapter(new_channel_idx)
    encoder.expand_channels(1)
    
    spikes_after = encoder.encode(seq)
    
    with torch.no_grad():
        _, _, u_pre_after = model(spikes_after)
        
    for t in range(18):
        diff = (u_pre_after[:, t, :] - u_pre_baseline[:, t, :]).abs().max().item()
        print(f"Time {t} max diff: {diff}")
        
    # Check intermediate calculations
    # Let's manually do one step of baseline vs after
    print("\n--- Manual step 0 ---")
    x_t_base = spikes_baseline[:, 0, :]
    x_t_after = spikes_after[:, 0, :]
    print(f"x_t diff: {(x_t_after[:, :3] - x_t_base).abs().max().item()}")
    
    # Check old weight vs new weight
    # Oh wait! We overwrote the model!
    
if __name__ == "__main__":
    test()
