import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from spiking_dreamcoder import SpikeTemporalEncoder, SpikingDreamCoder, StrictResonatorLayer, SemanticReadout, SleepEngine
from autonomous_miner import AutonomousMiner
from task_grounding_benchmark import GridWorldEnv, generate_dataset

def train_readout_batched(model, encoder, dataset, epochs=15, batch_size=64):
    optimizer = optim.Adam(model.readout.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()
    losses, accs = [], []
    
    for epoch in range(epochs):
        random.shuffle(dataset)
        total_loss, correct = 0, 0
        
        for i in range(0, len(dataset), batch_size):
            batch_data = dataset[i:i+batch_size]
            spikes_list = [encoder.encode(" ".join(seq)) for seq, label in batch_data]
            labels = torch.tensor([label for seq, label in batch_data], dtype=torch.long)
            
            # Pad sequences
            spikes_padded = pad_sequence(spikes_list, batch_first=True) # [batch, max_time, channels]
            
            optimizer.zero_grad()
            with torch.no_grad():
                _, u_pre_history = model.resonator(spikes_padded)
                
            # Extract terminal state for each sequence based on true length
            lengths = torch.tensor([s.shape[0] for s in spikes_list])
            terminal_states = u_pre_history[torch.arange(len(batch_data)), lengths - 1, :]
            
            logits = model.readout(terminal_states)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch_data)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            
        acc = correct / len(dataset)
        losses.append(total_loss / len(dataset))
        accs.append(acc)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {losses[-1]:.4f} | Acc: {acc*100:.2f}%")
        
    return losses, accs

def main():
    print("=== GRIDWORLD BENCHMARK (FAST BATCHED) ===")
    torch.manual_seed(42)
    random.seed(42)
    
    num_train, num_test = 500, 100
    seq_length = 60
    
    print(f"Генерация датасета (Length={seq_length})...")
    full_data = generate_dataset(num_train + num_test, length=seq_length)
    train_data, test_data = full_data[:num_train], full_data[num_train:]
    
    encoder_base = SpikeTemporalEncoder(channels=3, t_sym=6)
    model_base = SpikingDreamCoder(channels_in=3, hidden_size=32, num_classes=64)
    for p in model_base.resonator.parameters(): p.requires_grad = False
        
    print("\n--- Обучение BASELINE SNN ---")
    base_losses, base_accs = train_readout_batched(model_base, encoder_base, train_data, epochs=15)
    
    print("\n--- Обучение SPIKING DREAMCODER ---")
    encoder_dream = SpikeTemporalEncoder(channels=3, t_sym=6)
    model_dream = SpikingDreamCoder(channels_in=3, hidden_size=32, num_classes=64)
    
    miner = AutonomousMiner(encoder_dream, model_dream, threshold_k=50, lr=0.01)
    
    # Fast mining
    all_symbols = []
    for seq, _ in train_data: all_symbols.extend(seq)
    miner.stream_tokens(all_symbols)
    
    train_data_compressed = []
    for seq, label in train_data:
        comp_seq = miner.apply_bpe(seq)
        if len(comp_seq) > 0:
            train_data_compressed.append((comp_seq, label))
            
    print(f"\nДлина сырой строки: 60 | После сжатия (в среднем): {sum([len(s) for s, _ in train_data_compressed])/len(train_data_compressed):.1f}")
    
    for p in model_dream.resonator.parameters(): p.requires_grad = False
    dream_losses, dream_accs = train_readout_batched(model_dream, encoder_dream, train_data_compressed, epochs=15)
    
    plt.figure(figsize=(10, 5))
    plt.plot(base_accs, label="Baseline SNN", color='red')
    plt.plot(dream_accs, label="Spiking DreamCoder", color='green')
    plt.title("GridWorld 8x8: Navigation Benchmark (Length=60)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("benchmark_results_final.png")
    print("График сохранен в benchmark_results_final.png")

if __name__ == "__main__":
    main()
