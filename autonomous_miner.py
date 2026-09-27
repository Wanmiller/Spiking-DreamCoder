import torch
from collections import Counter
from spiking_dreamcoder import SpikeTemporalEncoder, SpikingDreamCoder, SleepEngine

class AutonomousMiner:
    """
    Автономный Майнер Паттернов (Unsupervised MDL / BPE).
    Анализирует непрерывный поток данных. При нахождении частых n-грамм (сжатие которых 
    уменьшает общую сложность по принципу MDL), инициирует "тихий час" (Sleep Phase),
    дистиллирует новый символ и переписывает историю.
    """
    def __init__(self, encoder, model, threshold_k=3, lr=0.01):
        self.encoder = encoder
        self.model = model
        self.sleep_engine = SleepEngine(encoder, model, threshold_k=threshold_k, lr=lr)
        self.threshold_k = threshold_k
        self.memory_stream = []
        self.bpe_rules = [] # Сохраняем историю замен (pair -> new_sym)
        
    def stream_tokens(self, tokens):
        """Добавляет токены в поток и запускает автономный анализ"""
        self.memory_stream.extend(tokens)
        self.process_stream()
        
    def process_stream(self):
        """
        Итеративный процесс BPE/MDL:
        1. Ищем самую частую пару.
        2. Если частота >= порога (MDL эвристика), уходим в сон.
        3. Заменяем пару на новый токен.
        4. Повторяем, пока есть пары для сжатия.
        """
        while len(self.memory_stream) >= 2:
            # 1. Считаем частоты биграмм
            bigrams = [(self.memory_stream[i], self.memory_stream[i+1]) for i in range(len(self.memory_stream) - 1)]
            counts = Counter(bigrams)
            
            if not counts:
                break
                
            best_pair, best_count = counts.most_common(1)[0]
            
            # 2. MDL порог: если сжатие не выгодно, останавливаемся
            if best_count < self.threshold_k:
                break
                
            print(f"\n[MDL TRIGGER] Обнаружен паттерн {best_pair} с частотой {best_count}. Инициация Сна...")
            
            # 3. Передаем в Sleep Engine
            self.sleep_engine.co_occurrence[best_pair] = best_count
            
            # Уходим в сон и дистиллируем
            # Для надежности увеличим rem_iterations (обычно 300 достаточно для сходимости)
            new_symbols = self.sleep_engine.sleep_phase(rem_iterations=300)
            
            if not new_symbols:
                print(f"[WARNING] Сон не привел к созданию символа для {best_pair}!")
                break
                
            new_sym_name, _ = new_symbols[0]
            
            # --- VALIDATION (Гетерогенная дистилляция) ---
            with torch.no_grad():
                teacher_seq = f"{best_pair[0]} {best_pair[1]}"
                t_spikes = self.encoder.encode(teacher_seq)
                s_spikes = self.encoder.encode(new_sym_name)
                
                _, _, u_pre_t = self.model(t_spikes)
                _, _, u_pre_s = self.model(s_spikes)
                
                term_t = u_pre_t[:, -1, :]
                term_s = u_pre_s[:, -1, :]
                
                # Центрируем векторы перед сравнением
                term_t_c = term_t - term_t.mean(dim=1, keepdim=True)
                term_s_c = term_s - term_s.mean(dim=1, keepdim=True)
                
                sim = torch.nn.functional.cosine_similarity(term_t_c, term_s_c).item()
                print(f"[VALIDATION] Гетерогенная дистилляция '{teacher_seq}' -> {new_sym_name}. Cosine Similarity: {sim:.4f}")
            # --------------------------------------------
            
            print(f"[MDL REWRITE] Перезапись памяти: замена {best_pair} на {new_sym_name}...")
            
            self.bpe_rules.append((best_pair, new_sym_name))
            
            # 4. Переписываем поток (BPE replacement)
            new_stream = []
            i = 0
            while i < len(self.memory_stream):
                if i < len(self.memory_stream) - 1 and (self.memory_stream[i], self.memory_stream[i+1]) == best_pair:
                    new_stream.append(new_sym_name)
                    i += 2
                else:
                    new_stream.append(self.memory_stream[i])
                    i += 1
                    
            self.memory_stream = new_stream
            print(f"[MDL UPDATE] Текущая длина памяти: {len(self.memory_stream)} токенов.")

    def apply_bpe(self, sequence):
        """Применяет найденные правила BPE к новой последовательности"""
        seq = sequence.copy()
        for pair, new_sym in self.bpe_rules:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i+1]) == pair:
                    new_seq.append(new_sym)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq = new_seq
        return seq

def test_autonomous_miner():
    print("Инициализация Автономного Майнера...")
    torch.manual_seed(123)
    
    encoder = SpikeTemporalEncoder(t_sym=6, channels=3)
    model = SpikingDreamCoder(channels_in=3, hidden_size=16, num_classes=2)
    
    miner = AutonomousMiner(encoder, model, threshold_k=3, lr=0.01)
    
    # Симулируем непрерывный поток, в котором есть скрытая иерархия:
    # Базовые абстракции: "A A"
    # Мета-абстракции: "A A A A" (которая будет сжата сначала в B0 B0, а затем в B1)
    
    stream = "0 1 A A 0 A A 1 0 1 A A 1 0 A A A A A A 1 0".split()
    
    print(f"Входящий поток: {' '.join(stream)}")
    miner.stream_tokens(stream)
    
    print(f"\nФинальное состояние памяти: {' '.join(miner.memory_stream)}")
    
if __name__ == '__main__':
    test_autonomous_miner()
