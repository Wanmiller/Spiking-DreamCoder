import torch
import torch.nn as nn
from spiking_dreamcoder import SpikingDreamCoder, SpikeTemporalEncoder, SleepEngine

def run_strict_validation():
    torch.manual_seed(123)
    print("=== STRICT VALIDATION & NEGATIVE CHECKS ===")
    
    # Инициализация
    encoder = SpikeTemporalEncoder()
    model = SpikingDreamCoder(channels_in=3, hidden_size=16, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Фиксируем состояние модели до сна
    model_state_before = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Тестовые данные для контроля забывания (Catastrophic Forgetting)
    control_seqs = ["1 0 1", "0 1 0", "A 1 0"]
    print("\n1. SANITY CHECK: Baseline (До сна)")
    baseline_hidden = {}
    with torch.no_grad():
        for seq in control_seqs:
            spikes = encoder.encode(seq)
            _, spikes_out, u_pre = model(spikes)
            baseline_hidden[seq] = u_pre[:, -1, :] # Terminal pre-reset state
            # Проверка на вырождение (Dead Neurons / Representation Collapse)
            spike_sparsity = (spikes_out == 0.0).float().mean().item()
            print(f"Seq '{seq}' | True Spike Sparsity: {spike_sparsity:.4f} | Mean U: {u_pre.mean().item():.4f}")
            if spike_sparsity > 0.98:
                print(f"[WARNING] SNN is mostly dead for {seq}. Sparsity > 98%.")

    print("\n1.5 SANITY CHECK: Effective Dimensionality (Participation Ratio)")
    H_list = []
    with torch.no_grad():
        for _ in range(50):
            # Генерируем 50 случайных входных спайковых паттернов
            rand_spikes = torch.randint(0, 2, (1, 6, encoder.channels)).float()
            _, _, u_pre = model(rand_spikes)
            H_list.append(u_pre[0, -1, :])
            
        H = torch.stack(H_list, dim=0) # [50, 16]
        H_centered = H - H.mean(dim=0)
        U, S, V = torch.linalg.svd(H_centered)
        eigenvalues = S ** 2
        participation_ratio = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
        print(f"Эффективная размерность пространства (из 16): {participation_ratio.item():.2f}")

    # Имитируем бодрствование и накопление статистики
    sleep_engine = SleepEngine(encoder, model, threshold_k=1, lr=0.01)
    sleep_engine.observe(['A', 'A'])
    
    print("\n2. EXECUTING REM SLEEP (Distillation)")
    # Проводим дистилляцию
    sleep_engine.sleep_phase(rem_iterations=300)
    
    print("\n3. SANITY CHECK: Catastrophic Forgetting (После сна)")
    forgetting_errors = []
    with torch.no_grad():
        for seq in control_seqs:
            spikes = encoder.encode(seq)
            _, _, u_pre = model(spikes)
            new_hidden_term = u_pre[:, -1, :]
            
            # Насколько сильно уплыли старые концепты из-за обновления общих весов?
            mse_drift = nn.functional.mse_loss(baseline_hidden[seq], new_hidden_term).item()
            cos_sim = nn.functional.cosine_similarity(baseline_hidden[seq], new_hidden_term).item()
            print(f"Seq '{seq}' | MSE Drift: {mse_drift:.6f} | Cosine Sim: {cos_sim:.4f}")
            forgetting_errors.append(mse_drift)
            
    avg_drift = sum(forgetting_errors) / len(forgetting_errors)
    if avg_drift > 1e-3:
        print(f"[CRITICAL FLAW] Catastrophic forgetting detected. Average MSE drift: {avg_drift:.6f}. Общие веса сети были разрушены во время дистилляции абстракции.")

    print("\n4. SANITY CHECK: Semantic Match vs Negative Control")
    with torch.no_grad():
        # Target matching
        target_spikes = encoder.encode("A A")
        _, _, u_pre_target = model(target_spikes)
        target_term = u_pre_target[:, -1, :]
        
        # Student matching
        student_spikes = encoder.encode("B0")
        _, _, u_pre_student = model(student_spikes)
        student_term = u_pre_student[:, -1, :]
        
        # Negative control (случайная последовательность, которая НЕ должна совпадать)
        negative_spikes = encoder.encode("1 0")
        _, _, u_pre_neg = model(negative_spikes)
        negative_term = u_pre_neg[:, -1, :]
        
        # Очищенное косинусное сходство (с вычитанием фона покоя)
        target_centered = target_term - target_term.mean(dim=1, keepdim=True)
        student_centered = student_term - student_term.mean(dim=1, keepdim=True)
        negative_centered = negative_term - negative_term.mean(dim=1, keepdim=True)
        
        match_sim = nn.functional.cosine_similarity(target_centered, student_centered).item()
        neg_sim = nn.functional.cosine_similarity(target_centered, negative_centered).item()
        
        print(f"Match Similarity (A A vs B0, centered): {match_sim:.4f}")
        print(f"Negative Similarity (A A vs 1 0, centered): {neg_sim:.4f}")
        
        if neg_sim > 0.9:
            print(f"[CRITICAL FLAW] Representation Collapse / Trivial Solution. Negative control ('1 0') is highly similar ({neg_sim:.4f}) to target ('A A'). The network is just outputting a constant or highly overlapping mean vector for everything.")
            
        print("\n5. OOD COMPOSITIONALITY & GENERALIZATION CHECK")
        with torch.no_grad():
            # Test 1: Replacement in context (X1 vs X2)
            seq_x1 = '0 A A 1'
            seq_x2 = '0 B0 1'
            
            spikes_x1 = encoder.encode(seq_x1)
            spikes_x2 = encoder.encode(seq_x2)
            
            _, _, u_x1 = model(spikes_x1)
            _, _, u_x2 = model(spikes_x2)
            
            term_x1 = u_x1[:, -1, :]
            term_x2 = u_x2[:, -1, :]
            
            term_x1_cent = term_x1 - term_x1.mean(dim=1, keepdim=True)
            term_x2_cent = term_x2 - term_x2.mean(dim=1, keepdim=True)
            
            sim_x1_x2 = nn.functional.cosine_similarity(term_x1_cent, term_x2_cent).item()
            print(f"Context Replacement ('0 A A 1' vs '0 B0 1') Sim: {sim_x1_x2:.4f}")
            
            # Test 2: Self-Composition / Recursion (X3 vs X4)
            seq_x4 = 'A A A A'
            seq_x3 = 'B0 B0'
            
            spikes_x4 = encoder.encode(seq_x4)
            spikes_x3 = encoder.encode(seq_x3)
            
            _, _, u_x4 = model(spikes_x4)
            _, _, u_x3 = model(spikes_x3)
            
            term_x4 = u_x4[:, -1, :]
            term_x3 = u_x3[:, -1, :]
            
            term_x4_cent = term_x4 - term_x4.mean(dim=1, keepdim=True)
            term_x3_cent = term_x3 - term_x3.mean(dim=1, keepdim=True)
            
            sim_x3_x4 = nn.functional.cosine_similarity(term_x4_cent, term_x3_cent).item()
            print(f"Self-Composition ('A A A A' vs 'B0 B0') Sim: {sim_x3_x4:.4f}")
            
            if sim_x1_x2 < 0.95 or sim_x3_x4 < 0.95:
                print("[WARNING] Generalization failed! The concept B0 works in isolation but does not compose perfectly in longer contexts.")
            else:
                print("[SUCCESS] Concept B0 is perfectly compositional and context-independent!")
                
        # Автоматический барьер качества (Sanity Gate):
        assert avg_drift == 0.0, f"FAIL: Забывание! Drift={avg_drift}"
        assert match_sim > 0.93, f"FAIL: B0 не выучил семантику! Sim={match_sim:.4f}"
        assert neg_sim < 0.45, f"FAIL: Коллапс пространства! Neg Sim={neg_sim:.4f}"
        assert (match_sim - neg_sim) > 0.50, "FAIL: Недостаточная избирательность!"
        print("\n>>> ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ: Код математически чист. <<<")

if __name__ == "__main__":
    run_strict_validation()
