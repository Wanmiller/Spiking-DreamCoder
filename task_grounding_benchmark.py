import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from spiking_dreamcoder import SpikeTemporalEncoder, SpikingDreamCoder, StrictResonatorLayer, SemanticReadout, SleepEngine
from autonomous_miner import AutonomousMiner

class GridWorldEnv:
    """Симуляция 8x8 лабиринта"""
    def __init__(self):
        self.size = 8
        
    def execute_path(self, path_symbols):
        # 0: East, 1: South, 2: West, 3: North
        x, y = 0, 0
        direction = 0 
        
        for sym in path_symbols:
            if sym == 'A': # Шаг вперед
                if direction == 0: x = (x + 1) % self.size
                elif direction == 1: y = (y + 1) % self.size
                elif direction == 2: x = (x - 1) % self.size
                elif direction == 3: y = (y - 1) % self.size
            elif sym == '1': # Поворот направо
                direction = (direction + 1) % 4
            elif sym == '0': # Стоп
                pass # Ничего не делаем
                
        return y * self.size + x # Класс от 0 до 63

def generate_dataset(num_samples=1000, length=30):
    env = GridWorldEnv()
    dataset = []
    
    # Генерация осмысленных путей (с макросами)
    # Макросы: "Длинный шаг вперед" (A A), "Зигзаг" (A 1 A)
    templates = [
        ['A', 'A', 'A'],
        ['A', 'A', '1'],
        ['1', 'A', 'A'],
        ['0']
    ]
    
    for _ in range(num_samples):
        seq = []
        while len(seq) < length:
            t = random.choice(templates)
            seq.extend(t)
        
        # Обрезаем точно до нужной длины
        seq = seq[:length]
        
        # Обязательно заканчиваем '0'
        seq[-1] = '0'
        
        label = env.execute_path(seq)
        dataset.append((seq, label))
        
    return dataset

def train_readout(model, encoder, dataset, epochs=15):
    """Тренировка только Readout слоя (Linear Probing)"""
    optimizer = optim.Adam(model.readout.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    losses = []
    accs = []
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        # Перемешиваем
        random.shuffle(dataset)
        
        for seq, label in dataset: # Батч-сайз 1 для SNN (можно векторизовать, но для демо хватит 1)
            optimizer.zero_grad()
            
            # Вычисляем состояние с замороженным резонатором, чтобы избежать багов с автоградом LoRA
            with torch.no_grad():
                spikes = encoder.encode(" ".join(seq))
                _, u_pre_history = model.resonator(spikes)
            
            # Обучаем только линейный слой
            logits = model.readout(u_pre_history)
            
            target = torch.tensor([label])
            loss = criterion(logits, target)
            loss.backward()
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1).item()
            if pred == label:
                correct += 1
                
        acc = correct / len(dataset)
        losses.append(total_loss / len(dataset))
        accs.append(acc)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {losses[-1]:.4f} | Acc: {acc*100:.2f}%")
        
    return losses, accs

def main():
    print("=== GRIDWORLD BENCHMARK ===")
    torch.manual_seed(42)
    random.seed(42)
    
    num_train = 500
    num_test = 100
    seq_length = 60 # Достаточно длинно, чтобы SNN начал забывать начало
    
    print(f"Генерация датасета (Length={seq_length})...")
    train_data = generate_dataset(num_train, seq_length)
    test_data = generate_dataset(num_test, seq_length)
    
    # ---------------------------------------------
    # 1. BASELINE: Классическая SNN (без майнера)
    # ---------------------------------------------
    print("\n--- Обучение BASELINE SNN ---")
    encoder_base = SpikeTemporalEncoder(t_sym=6, channels=3)
    model_base = SpikingDreamCoder(channels_in=3, hidden_size=32, num_classes=64)
    
    # Замораживаем резонатор (SNN Reservoir Computing / Liquid State Machine подход)
    for p in model_base.resonator.parameters():
        p.requires_grad = False
        
    base_losses, base_accs = train_readout(model_base, encoder_base, train_data, epochs=15)
    
    # ---------------------------------------------
    # 2. SPIKING DREAMCODER: С майнером
    # ---------------------------------------------
    print("\n--- Обучение SPIKING DREAMCODER ---")
    encoder_dream = SpikeTemporalEncoder(t_sym=6, channels=3)
    model_dream = SpikingDreamCoder(channels_in=3, hidden_size=32, num_classes=64)
    
    # Запускаем майнер на всем тренировочном датасете, слитом в одну строку
    miner = AutonomousMiner(encoder_dream, model_dream, threshold_k=50, lr=0.01)
    
    all_symbols = []
    for seq, _ in train_data:
        all_symbols.extend(seq)
        
    print("Запуск Автономного Майнера на корпусе...")
    miner.stream_tokens(all_symbols)
    
    # Теперь нужно сжать датасет, используя найденный словарь
    # Простейшая реализация: пропускаем каждую последовательность через ту же логику BPE
    def compress_sequence(seq, memory_stream, threshold_k):
        # Поскольку miner уже нашел словарь в memory_stream, 
        # нам нужно просто переиспользовать его (в данном демо мы просто 
        # соберем BPE-замены, но проще пере-прогнать майнер на тестовых данных
        # с нулевым lr, либо просто взять тот же алгоритм на seq).
        # Для чистоты эксперимента, применим BPE-слияния, которые были найдены.
        # В miner-е это можно было бы вынести в метод apply_bpe, но пока 
        # мы просто повторим замены:
        return seq # (TODO: реализуем функцию сжатия ниже)
        
    # В miner мы уже сжали весь корпус в miner.memory_stream!
    # Разобьем miner.memory_stream обратно на последовательности.
    # Так как мы склеивали весь корпус в один массив, это нетривиально (символ '0' был концом).
    # Давайте напишем локальный компрессор на основе miner.sleep_engine.co_occurrence
    
    print("Сжатие датасета...")
    train_data_compressed = []
    for seq, label in train_data:
        compressed_seq = miner.apply_bpe(seq)
        train_data_compressed.append((compressed_seq, label))
        
    test_data_compressed = []
    for seq, label in test_data:
        compressed_seq = miner.apply_bpe(seq)
        test_data_compressed.append((compressed_seq, label))
        
    print(f"Пример оригинальной длины: {len(train_data[0][0])}")
    print(f"Пример сжатой длины: {len(train_data_compressed[0][0])}")
    
    # Замораживаем резонатор
    for p in model_dream.resonator.parameters():
        p.requires_grad = False
        
    dream_losses, dream_accs = train_readout(model_dream, encoder_dream, train_data_compressed, epochs=15)
    
    # ---------------------------------------------
    # 3. ВИЗУАЛИЗАЦИЯ И СРАВНЕНИЕ
    # ---------------------------------------------
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(base_losses, label='Baseline SNN (Raw)')
    plt.plot(dream_losses, label='Spiking DreamCoder (Compressed)')
    plt.title('Readout Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('CrossEntropy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(base_accs, label='Baseline SNN (Raw)')
    plt.plot(dream_accs, label='Spiking DreamCoder (Compressed)')
    plt.title('Readout Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    print("\n[УСПЕХ] Графики сохранены в benchmark_results.png")
    
    # Тестирование на тестовой выборке
    print("\n--- ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ ---")
    def evaluate(model, encoder, dataset):
        model.eval()
        correct = 0
        with torch.no_grad():
            for seq, label in dataset:
                spikes = encoder.encode(" ".join(seq))
                logits, _, _ = model(spikes)
                pred = torch.argmax(logits, dim=1).item()
                if pred == label:
                    correct += 1
        return correct / len(dataset)
        
    base_test_acc = evaluate(model_base, encoder_base, test_data)
    dream_test_acc = evaluate(model_dream, encoder_dream, test_data_compressed)
    
    print(f"Baseline SNN Test Accuracy: {base_test_acc * 100:.2f}%")
    print(f"Spiking DreamCoder Test Accuracy: {dream_test_acc * 100:.2f}%")

if __name__ == '__main__':
    main()
