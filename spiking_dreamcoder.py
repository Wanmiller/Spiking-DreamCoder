import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------
# Surrogate Gradient for Spiking Neural Networks
# ---------------------------------------------------------
class FastSigmoidSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, alpha=5.0):
        ctx.save_for_backward(u)
        ctx.alpha = alpha
        return (u > 0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (u,) = ctx.saved_tensors
        alpha = ctx.alpha
        # Градиент суррогата с более широким окном пропускания
        surrogate_grad = alpha / (1.0 + alpha * torch.abs(u)) ** 2
        return grad_output * surrogate_grad, None

def spike_fn(u):
    return FastSigmoidSurrogate.apply(u, 2.0)

# ---------------------------------------------------------
# Strict Resonator Layer
# ---------------------------------------------------------
class StrictResonatorLayer(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, decay: float = 0.75, threshold: float = 0.3, gamma_inhib: float = 0.4):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.decay = decay
        self.threshold = threshold
        self.gamma_inhib = gamma_inhib

        # Входные и рекуррентные проекции
        self.w_in = nn.Linear(in_features, hidden_features, bias=False)
        self.w_rec = nn.Linear(hidden_features, hidden_features, bias=False)
        
        # Нормализация с аффинными параметрами для сдвига к порогу
        self.norm = nn.LayerNorm(hidden_features, elementwise_affine=True)
        
        # Рекуррентные адаптеры (LoRA)
        self.lora_u = nn.ParameterDict()
        self.lora_v = nn.ParameterDict()

        self._init_weights()

    def _init_weights(self):
        # Входные веса - Kaiming Normal с повышенной дисперсией
        nn.init.kaiming_normal_(self.w_in.weight, a=0.2, nonlinearity='leaky_relu')
        # Рекуррентные веса - ортогональные с масштабированием
        nn.init.orthogonal_(self.w_rec.weight)
        with torch.no_grad():
            self.w_rec.weight.mul_(0.85)  # Спектральный радиус < 1
            # Жестко сдвигаем рабочую точку ближе к порогу
            nn.init.constant_(self.norm.bias, self.threshold)
            nn.init.constant_(self.norm.weight, 1.0)
    def expand_input_channels(self, num_new_channels=1):
        """Динамическое выделение новых концепт-нейронов (каналов)"""
        old_weight = self.w_in.weight.data
        out_f, in_f = old_weight.shape
        new_weight = torch.zeros(out_f, in_f + num_new_channels, device=old_weight.device)
        new_weight[:, :in_f] = old_weight
        
        # Инициализация новых синапсов
        nn.init.kaiming_normal_(new_weight[:, in_f:], a=0.2, nonlinearity='leaky_relu')
        
        self.w_in = nn.Linear(in_f + num_new_channels, out_f, bias=False)
        self.w_in.weight.data = new_weight
        self.in_features += num_new_channels
        return in_f # Индекс первого нового канала

    def add_lora_adapter(self, channel_idx: int, rank: int = 4):
        # LoRA rank adapters
        u = nn.Parameter(torch.randn(rank, self.hidden_features) * 0.1)
        v = nn.Parameter(torch.randn(self.hidden_features, rank) * 0.1)
        self.lora_u[str(channel_idx)] = u
        self.lora_v[str(channel_idx)] = v

    def forward(self, x_seq: torch.Tensor):
        batch_size, time_steps, _ = x_seq.shape
        device = x_seq.device

        u = torch.zeros(batch_size, self.hidden_features, device=device)
        s = torch.zeros(batch_size, self.hidden_features, device=device)

        spikes_history = []
        u_pre_history = []

        # -------------------------------------------------------------
        # Vectorized LoRA Initialization
        # -------------------------------------------------------------
        num_concepts = len(self.lora_u)
        has_lora = num_concepts > 0
        if has_lora:
            active_c_indices = torch.tensor([int(c) for c in self.lora_u.keys()], device=device)
            # Stack weights to [num_concepts, rank, hidden] and [num_concepts, hidden, rank]
            U = torch.stack(list(self.lora_u.values()), dim=0)
            V = torch.stack(list(self.lora_v.values()), dim=0)
            lora_traces = torch.zeros(batch_size, num_concepts, device=device)
            decay_trace = 0.6  # Быстрое затухание следа

        for t in range(time_steps):
            x_t = x_seq[:, t, :]
            
            # Базовый рекуррентный отклик
            rec_current = self.w_rec(s)
            
            # Активация адаптеров LoRA с использованием синаптического следа (Векторизованно)
            if has_lora:
                max_c = x_t.shape[1]
                valid_mask = active_c_indices < max_c
                valid_indices = active_c_indices[valid_mask]
                
                spikes_c = torch.zeros(batch_size, num_concepts, device=device)
                if len(valid_indices) > 0:
                    spikes_c[:, valid_mask] = x_t[:, valid_indices]
                
                # 1. Обновляем след присутствия символа:
                lora_traces = decay_trace * lora_traces + spikes_c
                
                # 2. Локальный контекст: активен ТОЛЬКО пока жив след спайка (порог 0.05)
                local_ctx = (lora_traces > 0.05).float() # [batch_size, num_concepts]
                
                # 3. LoRA применяется строго модулированная локальным следом
                # s: [batch_size, hidden]
                # V: [num_concepts, hidden, rank]
                # U: [num_concepts, rank, hidden]
                s_V = torch.einsum('bh,chr->bcr', s, V)
                s_V_U = torch.einsum('bcr,crk->bck', s_V, U)
                
                # Взвешиваем и суммируем
                lora_stim_total = torch.sum(local_ctx.unsqueeze(2) * s_V_U, dim=1) # [batch_size, hidden]
                rec_current = rec_current + lora_stim_total
                
            # Ток = входной сигнал + рекуррентный отклик от прошлых спайков + LoRA
            current = self.w_in(x_t) + rec_current
            current = self.norm(current)
            
            # Латеральное торможение (Lateral Inhibition) применяется ПОСЛЕ нормализации, 
            # иначе LayerNorm съест равномерный сдвиг (mean shift).
            inhib = self.gamma_inhib * s.sum(dim=-1, keepdim=True)
            current = current - inhib

            # Динамика мембраны ДО сброса
            u_pre = self.decay * u + current
            u_pre_history.append(u_pre)

            # Генерация спайка через суррогат
            s = spike_fn(u_pre - self.threshold)
            spikes_history.append(s)

            # Сброс потенциала (Aggressive Soft reset)
            u = u_pre - s * 1.0

        return torch.stack(spikes_history, dim=1), torch.stack(u_pre_history, dim=1)

# ---------------------------------------------------------
# Module 1: Spike Temporal Encoder
# ---------------------------------------------------------
class SpikeTemporalEncoder:
    def __init__(self, t_sym=6, channels=3):
        self.t_sym = t_sym
        self.channels = channels
        self.symbol_dict = {
            '0': self._create_pattern([(1, 0)]),
            '1': self._create_pattern([(4, 2)]),
            'A': self._create_pattern([(0, 0), (2, 1), (4, 2)])
        }
        
    def _create_pattern(self, spikes):
        pattern = torch.zeros(self.t_sym, self.channels)
        for t, c in spikes:
            if t < self.t_sym and c < self.channels:
                pattern[t, c] = 1.0
        return pattern
        
    def add_symbol(self, symbol, pattern):
        if pattern.shape[0] != self.t_sym or pattern.shape[1] != self.channels:
            raise ValueError(f"Pattern must be of shape ({self.t_sym}, {self.channels})")
        self.symbol_dict[symbol] = pattern

    def expand_channels(self, num_new=1):
        """Обновление всех старых символов под новый размер тензора"""
        for sym in self.symbol_dict:
            old_pat = self.symbol_dict[sym]
            new_pat = torch.zeros(self.t_sym, self.channels + num_new)
            new_pat[:, :self.channels] = old_pat
            self.symbol_dict[sym] = new_pat
        self.channels += num_new
            
    def encode(self, string_input):
        symbols = string_input.strip().split()
        total_time = len(symbols) * self.t_sym
        
        tensor_out = torch.zeros(total_time, self.channels)
        
        for i, sym in enumerate(symbols):
            if sym in self.symbol_dict:
                tensor_out[i*self.t_sym : (i+1)*self.t_sym, :] = self.symbol_dict[sym]
                
        return tensor_out.unsqueeze(0)

# ---------------------------------------------------------
# Module 3: Semantic Readout
# ---------------------------------------------------------
class SemanticReadout(nn.Module):
    def __init__(self, hidden_size=16, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, u_pre_history):
        # Строгая интеграция: берем терминальное состояние до сброса
        # (вместо скомпрометированного mean по всем спайкам)
        terminal_state = u_pre_history[:, -1, :]
        logits = self.classifier(terminal_state)
        return logits

# ---------------------------------------------------------
# Full SNN Architecture
# ---------------------------------------------------------
class SpikingDreamCoder(nn.Module):
    def __init__(self, channels_in=3, hidden_size=16, num_classes=2):
        super().__init__()
        self.resonator = StrictResonatorLayer(in_features=channels_in, hidden_features=hidden_size)
        self.readout = SemanticReadout(hidden_size, num_classes)
        
    def forward(self, spike_seq):
        hidden_spikes, u_pre_history = self.resonator(spike_seq)
        logits = self.readout(u_pre_history)
        return logits, hidden_spikes, u_pre_history

# ---------------------------------------------------------
# Module 4: Contrastive Sleep Engine
# ---------------------------------------------------------
class SleepEngine:
    def __init__(self, encoder, model, threshold_k=3, lr=0.05, margin=1.0, lambda_neg=0.5):
        self.encoder = encoder
        self.model = model
        self.threshold_k = threshold_k
        self.co_occurrence = {}
        self.new_symbol_count = 0
        self.lr = lr
        self.margin = margin
        self.lambda_neg = lambda_neg
        self.writer = SummaryWriter('runs/spiking_dreamcoder_strict')
        
    def observe(self, symbol_seq):
        for i in range(len(symbol_seq) - 1):
            pair = (symbol_seq[i], symbol_seq[i+1])
            if pair not in self.co_occurrence:
                self.co_occurrence[pair] = 0
            self.co_occurrence[pair] += 1
            
    def sleep_phase(self, rem_iterations=50):
        new_symbols_created = []
        candidates = list(self.co_occurrence.items())
        
        for pair, count in candidates:
            if count >= self.threshold_k:
                sym1, sym2 = pair
                new_sym_name = f'B{self.new_symbol_count}'
                self.new_symbol_count += 1
                
                # 1. Аллокация Концепт-Нейрона (Новый канал)
                new_channel_idx = self.model.resonator.expand_input_channels(1)
                self.model.resonator.add_lora_adapter(new_channel_idx)
                self.encoder.expand_channels(1)
                
                # Символ B0 - это дуплет спайков на новом канале (t=0 и t=2)
                b0_spikes = torch.zeros(self.encoder.t_sym, self.encoder.channels)
                b0_spikes[0, new_channel_idx] = 1.0
                if self.encoder.t_sym > 2:
                    b0_spikes[2, new_channel_idx] = 1.0
                self.encoder.add_symbol(new_sym_name, b0_spikes)
                
                # 2. Подготовка Целей
                teacher_seq = f"{sym1} {sym2}"
                negative_seq = "1 0"  # Негативный контроль
                
                teacher_spikes = self.encoder.encode(teacher_seq)
                negative_spikes = self.encoder.encode(negative_seq)
                student_spikes = self.encoder.encode(new_sym_name)
                
                with torch.no_grad():
                    _, u_teacher_hist = self.model.resonator(teacher_spikes)
                    _, u_neg_hist = self.model.resonator(negative_spikes)
                    
                    target_state = u_teacher_hist[:, -1, :].detach()
                    negative_state = u_neg_hist[:, -1, :].detach()
                
                # 3. Замораживаем все параметры, кроме матрицы w_in
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.resonator.w_in.weight.requires_grad = True
                
                # Сохраняем точные копии старых весов для хард-ресета (защита от дрейфа)
                frozen_w_in = self.model.resonator.w_in.weight.data[:, :-1].clone()
                frozen_w_rec = self.model.resonator.w_rec.weight.data.clone()
                frozen_norm_w = self.model.resonator.norm.weight.data.clone()
                frozen_norm_b = self.model.resonator.norm.bias.data.clone()
                
                # Оптимизатор обучает веса входного слоя и новые вектора LoRA
                u_param = self.model.resonator.lora_u[str(new_channel_idx)]
                v_param = self.model.resonator.lora_v[str(new_channel_idx)]
                u_param.requires_grad = True
                v_param.requires_grad = True
                
                optimizer = torch.optim.Adam([
                    self.model.resonator.w_in.weight,
                    u_param,
                    v_param
                ], lr=self.lr)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=rem_iterations, eta_min=1e-4)
                
                print(f"\n--- [Strict Sleep Phase] Allocating Concept Neuron for '{sym1}' + '{sym2}' -> '{new_sym_name}' ---")
                
                loss_history = []
                best_loss = float('inf')
                best_weights_w_in = None
                best_weights_u = None
                best_weights_v = None
                
                for step in range(rem_iterations):
                    optimizer.zero_grad()
                    
                    # Прогон через резонатор
                    _, u_stud_hist = self.model.resonator(student_spikes)
                    pred_state = u_stud_hist[:, -1, :]
                    
                    loss_align = F.mse_loss(pred_state, target_state)
                    dist_neg = torch.norm(pred_state - negative_state, p=2)
                    loss_contrast = torch.relu(self.margin - dist_neg)
                    
                    loss_total = loss_align + self.lambda_neg * loss_contrast
                    loss_total.backward()
                    
                    # МАСКИРОВАНИЕ ГРАДИЕНТОВ: Обнуляем градиенты для старых каналов, 
                    # чтобы они не забыли свои концепты. Обновляется ТОЛЬКО последний столбец.
                    self.model.resonator.w_in.weight.grad[:, :-1] = 0.0
                    
                    # Gradient Clipping для предотвращения выбросов
                    torch.nn.utils.clip_grad_norm_([self.model.resonator.w_in.weight, u_param, v_param], max_norm=1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    
                    loss_val = loss_total.item()
                    loss_history.append(loss_val)
                    
                    # Сохраняем лучшие веса на "дне" минимума
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_weights_w_in = self.model.resonator.w_in.weight.data.clone()
                        best_weights_u = u_param.data.clone()
                        best_weights_v = v_param.data.clone()
                    
                    # Логирование в TensorBoard
                    self.writer.add_scalar(f"Distillation_{new_sym_name}/Total_Loss", loss_val, step)
                    self.writer.add_scalar(f"Distillation_{new_sym_name}/Align_MSE", loss_align.item(), step)
                    self.writer.add_scalar(f"Distillation_{new_sym_name}/Negative_Distance", dist_neg.item(), step)
                        
                    if (step + 1) % 10 == 0 or step == 0:
                        print(f"Step {step+1:02d} | Total Loss: {loss_val:.5f} | Align MSE: {loss_align.item():.5f} | Neg Dist: {dist_neg.item():.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")
                
                # Фиксируем лучшие веса
                self.model.resonator.w_in.weight.data.copy_(best_weights_w_in)
                u_param.data.copy_(best_weights_u)
                v_param.data.copy_(best_weights_v)
                
                # Жестко восстанавливаем замороженные веса (защита от невидимого дрейфа)
                w_in_drift = torch.max(torch.abs(self.model.resonator.w_in.weight.data[:, :-1] - frozen_w_in)).item()
                w_rec_drift = torch.max(torch.abs(self.model.resonator.w_rec.weight.data - frozen_w_rec)).item()
                print(f"DEBUG Drift before restore -> w_in: {w_in_drift:.8f}, w_rec: {w_rec_drift:.8f}")
                
                self.model.resonator.w_in.weight.data[:, :-1] = frozen_w_in
                self.model.resonator.w_rec.weight.data.copy_(frozen_w_rec)
                self.model.resonator.norm.weight.data.copy_(frozen_norm_w)
                self.model.resonator.norm.bias.data.copy_(frozen_norm_b)
                
                print(f"Зафиксирован абсолютный минимум Loss: {best_loss:.4f}")
                        
                new_symbols_created.append((new_sym_name, pair))
                
                self.co_occurrence[pair] = 0
                
        return new_symbols_created

# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def plot_raster(spikes, title="Spike Raster"):
    time, channels = spikes.shape
    plt.figure(figsize=(8, 4))
    for c in range(channels):
        spike_times = torch.nonzero(spikes[:, c]).flatten().numpy()
        if len(spike_times) > 0:
            plt.scatter(spike_times, [c]*len(spike_times), marker='|', color='black', s=150)
    plt.yticks(range(channels), [f"Ch {c}" for c in range(channels)])
    plt.xlim(-1, time)
    plt.xlabel("Time (ticks)")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

if __name__ == "__main__":
    print("Initializing Spiking DreamCoder v0.2 (Strict)...")
    encoder = SpikeTemporalEncoder()
    model = SpikingDreamCoder(channels_in=3, hidden_size=16, num_classes=2)
    sleep_engine = SleepEngine(encoder, model, threshold_k=2)
    
    # 1. Wake Phase
    print("\n--- [Wake Phase] ---")
    seqs = ["A A 1 0", "A A 0 1", "A A A"]
    for seq in seqs:
        spikes = encoder.encode(seq)
        logits, _, _ = model(spikes)
        sleep_engine.observe(seq.split())
        print(f"Observed '{seq}'")

    # 2. Sleep Phase
    new_symbols = sleep_engine.sleep_phase(rem_iterations=50)
