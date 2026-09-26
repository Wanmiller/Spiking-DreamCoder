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
        baseline_out = u_pre_baseline[:, -1, :]
        
    print(f"Spikes baseline shape: {spikes_baseline.shape}")
    print(f"w_in weight shape baseline: {model.resonator.w_in.weight.shape}")
    
    # 1. Expand channels
    new_channel_idx = model.resonator.expand_input_channels(1)
    model.resonator.add_lora_adapter(new_channel_idx)
    encoder.expand_channels(1)
    
    spikes_after = encoder.encode(seq)
    
    print(f"Spikes after shape: {spikes_after.shape}")
    print(f"w_in weight shape after: {model.resonator.w_in.weight.shape}")
    
    # Check if first 3 channels of spikes are identical
    diff_spikes = (spikes_after[:, :, :3] - spikes_baseline).abs().max().item()
    print(f"Diff spikes (first 3 channels): {diff_spikes}")
    
    # Check if 4th channel is 0
    max_4th_channel = spikes_after[:, :, 3].abs().max().item()
    print(f"Max 4th channel: {max_4th_channel}")
    
    with torch.no_grad():
        _, _, u_pre_after = model(spikes_after)
        after_out = u_pre_after[:, -1, :]
        
    diff_out = (after_out - baseline_out).abs().max().item()
    print(f"Diff output BEFORE training: {diff_out}")
    
    # Simulate training by corrupting the 4th column of w_in
    model.resonator.w_in.weight.data[:, 3] = torch.randn(16) * 5.0
    
    with torch.no_grad():
        _, _, u_pre_train = model(spikes_after)
        train_out = u_pre_train[:, -1, :]
        
    diff_out_train = (train_out - baseline_out).abs().max().item()
    print(f"Diff output AFTER training 4th column: {diff_out_train}")

if __name__ == "__main__":
    test()
