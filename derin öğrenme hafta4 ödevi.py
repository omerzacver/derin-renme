import time
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# =====================================================================
# 1. TEMEL AYARLAR VE VERİ KÜMESİ
# =====================================================================
# Okul numaranın son iki hanesini buraya yaz:
SEED = 22  # Örnek: okul numaran 22 ile bitiyorsa 22 yap

np.random.seed(SEED)
tf.random.set_seed(SEED)

# Veri Kümesi: y = 1 + sin(pi * p / 4), p in [-2, 2]
p = np.linspace(-2, 2, 400)
y = 1 + np.sin(np.pi * p / 4)
X_train = p.reshape(-1, 1)
y_train = y.reshape(-1, 1)

os.makedirs("grafikler", exist_ok=True)
print("Veri kümesi oluşturuldu. Deneyler başlatılıyor...\n")


# =====================================================================
# 2. GERÇEK R2 SKORUNU EPOCH BOYUNCA KAYDEDEN CALLBACK
# (Arkadaşının sahte düz çizgi hatasını çözen kısım)
# =====================================================================
class R2HistoryCallback(keras.callbacks.Callback):
    def __init__(self, x_data, y_data):
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.r2_scores = []

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.x_data, verbose=0)
        score = r2_score(self.y_data, preds)
        self.r2_scores.append(score)


# =====================================================================
# 3. GENEL MODEL EĞİTİM VE TEST FONKSİYONU
# =====================================================================
def train_and_evaluate(build_model_fn, optimizer, epochs=300, batch_size=32):
    model = build_model_fn()
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    r2_cb = R2HistoryCallback(X_train, y_train)
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[r2_cb]
    )
    elapsed = time.time() - start_time
    
    y_pred = model.predict(X_train, verbose=0)
    final_rmse = np.sqrt(history.history['loss'][-1])
    final_r2 = r2_score(y_train, y_pred)
    rmse_curve = np.sqrt(history.history['loss'])
    
    return {
        "model": model,
        "rmse": final_rmse,
        "r2": final_r2,
        "time": elapsed,
        "rmse_curve": rmse_curve,
        "r2_curve": r2_cb.r2_scores,
        "y_pred": y_pred
    }


def plot_experiment(results, title_prefix, best_key, filename):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Loss / RMSE Log Eğrileri
    for key, res in results.items():
        axes[0].plot(res["rmse_curve"], linewidth=1.8, label=str(key))
    axes[0].set_yscale('log')
    axes[0].set_title(f"{title_prefix} / RMSE Error (Log)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (RMSE)")
    axes[0].legend()
    axes[0].grid(True, which="both", ls="--", alpha=0.6)

    # Panel 2: En İyi Modelin Hedef vs Çıkış Eğrisi
    best_res = results[best_key]
    axes[1].plot(p, y, label='Hedef (Target)', color='midnightblue', linewidth=2)
    axes[1].plot(p, best_res["y_pred"], label=f'Ağ Çıkışı ({best_key})', color='orangered', linestyle='--', linewidth=2)
    axes[1].set_title(f"Hedef vs Ağ Çıkışı ({best_key})")
    axes[1].set_xlabel("Giriş (p)")
    axes[1].set_ylabel("Çıkış (y)")
    axes[1].legend()
    axes[1].grid(True, ls="--", alpha=0.6)

    # Panel 3: Gerçek Epoch / R2 Skoru Eğrisi
    axes[2].plot(range(1, len(best_res["r2_curve"]) + 1), best_res["r2_curve"], color='forestgreen', linewidth=2)
    axes[2].set_title(f"Epoch / R2 Skoru ({best_key})")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("R2 Score")
    axes[2].grid(True, ls="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"grafikler/{filename}.png", dpi=200)
    plt.show()


# =====================================================================
# ADIM 1: AKTİVASYON FONKSİYONLARI DENEYİ
# =====================================================================
print("=== 1. AKTİVASYON FONKSİYONLARI DENEYİ ===")
activations = ['linear', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'silu']
results_act = {}

print(f"{'Aktivasyon':<15} | {'RMSE':<10} | {'R2':<10} | {'Süre (sn)':<10}")
print("-" * 55)

for act in activations:
    def build_act_model(a=act):
        return keras.Sequential([
            layers.Input(shape=(1,)),
            layers.Dense(2, activation=a, kernel_initializer=keras.initializers.RandomNormal(seed=SEED)),
            layers.Dense(1, activation='linear', kernel_initializer=keras.initializers.RandomNormal(seed=SEED))
        ])
    res = train_and_evaluate(build_act_model, optimizer=keras.optimizers.SGD(learning_rate=0.1))
    results_act[act] = res
    print(f"{act:<15} | {res['rmse']:<10.5f} | {res['r2']:<10.5f} | {res['time']:<10.2f}")

plot_experiment(results_act, "Aktivasyon", "sigmoid", "1_aktivasyon_deneyi")


# =====================================================================
# ADIM 2: ÖĞRENME ORANI (LEARNING RATE) DENEYİ
# =====================================================================
print("\n=== 2. ÖĞRENME ORANI DENEYİ ===")
learning_rates = [0.1, 0.01, 0.001]
results_lr = {}

print(f"{'Öğrenme Oranı':<15} | {'RMSE':<10} | {'R2':<10} | {'Süre (sn)':<10}")
print("-" * 55)

for lr in learning_rates:
    def build_lr_model():
        return keras.Sequential([
            layers.Input(shape=(1,)),
            layers.Dense(2, activation='sigmoid', kernel_initializer=keras.initializers.RandomNormal(seed=SEED)),
            layers.Dense(1, activation='linear', kernel_initializer=keras.initializers.RandomNormal(seed=SEED))
        ])
    res = train_and_evaluate(build_lr_model, optimizer=keras.optimizers.SGD(learning_rate=lr))
    results_lr[f"LR={lr}"] = res
    print(f"{str(lr):<15} | {res['rmse']:<10.5f} | {res['r2']:<10.5f} | {res['time']:<10.2f}")

plot_experiment(results_lr, "Öğrenme Oranı", "LR=0.1", "2_ogrenme_orani_deneyi")


# =====================================================================
# ADIM 3: OPTİMİZASYON YÖNTEMLERİ DENEYİ
# =====================================================================
print("\n=== 3. OPTİMİZASYON YÖNTEMLERİ DENEYİ ===")
opt_dict = {
    'SGD': keras.optimizers.SGD(learning_rate=0.01),
    'RMSprop': keras.optimizers.RMSprop(learning_rate=0.01),
    'ADAM': keras.optimizers.Adam(learning_rate=0.01)
}
results_opt = {}

print(f"{'Optimizasyon':<15} | {'RMSE':<10} | {'R2':<10} | {'Süre (sn)':<10}")
print("-" * 55)

for name, opt in opt_dict.items():
    def build_opt_model():
        return keras.Sequential([
            layers.Input(shape=(1,)),
            layers.Dense(2, activation='sigmoid', kernel_initializer=keras.initializers.RandomNormal(seed=SEED)),
            layers.Dense(1, activation='linear', kernel_initializer=keras.initializers.RandomNormal(seed=SEED))
        ])
    res = train_and_evaluate(build_opt_model, optimizer=opt)
    results_opt[name] = res
    print(f"{name:<15} | {res['rmse']:<10.5f} | {res['r2']:<10.5f} | {res['time']:<10.2f}")

plot_experiment(results_opt, "Optimizasyon", "ADAM", "3_optimizasyon_deneyi")


# =====================================================================
# ADIM 4: AĞIRLIK BAŞLANGIÇ DAĞILIMLARI DENEYİ
# =====================================================================
print("\n=== 4. AĞIRLIK BAŞLANGIÇ DAĞILIMLARI DENEYİ ===")
initializers = {
    'RandomNormal': keras.initializers.RandomNormal(seed=SEED),
    'RandomUniform': keras.initializers.RandomUniform(seed=SEED),
    'GlorotUniform': keras.initializers.GlorotUniform(seed=SEED),
    'HeNormal': keras.initializers.HeNormal(seed=SEED)
}
results_init = {}

print(f"{'Dağılım':<15} | {'RMSE':<10} | {'R2':<10} | {'Süre (sn)':<10}")
print("-" * 55)

for name, init in initializers.items():
    def build_init_model(i=init):
        return keras.Sequential([
            layers.Input(shape=(1,)),
            layers.Dense(2, activation='sigmoid', kernel_initializer=i),
            layers.Dense(1, activation='linear', kernel_initializer=i)
        ])
    res = train_and_evaluate(build_init_model, optimizer=keras.optimizers.Adam(learning_rate=0.01))
    results_init[name] = res
    print(f"{name:<15} | {res['rmse']:<10.5f} | {res['r2']:<10.5f} | {res['time']:<10.2f}")

plot_experiment(results_init, "Initializer", "RandomNormal", "4_initializer_deneyi")


# =====================================================================
# ADIM 5: GİZLİ KATMAN SAYISI DENEYİ
# =====================================================================
print("\n=== 5. GİZLİ KATMAN SAYISI DENEYİ ===")
layer_counts = [1, 2, 5, 10, 20]
results_layers = {}

print(f"{'Katman Sayısı':<15} | {'RMSE':<10} | {'R2':<10} | {'Süre (sn)':<10}")
print("-" * 55)

for count in layer_counts:
    def build_deep_model(c=count):
        seq = [layers.Input(shape=(1,))]
        for _ in range(c):
            seq.append(layers.Dense(2, activation='sigmoid', kernel_initializer=keras.initializers.RandomNormal(seed=SEED)))
        seq.append(layers.Dense(1, activation='linear', kernel_initializer=keras.initializers.RandomNormal(seed=SEED)))
        return keras.Sequential(seq)
    res = train_and_evaluate(build_deep_model, optimizer=keras.optimizers.Adam(learning_rate=0.01))
    results_layers[f"{count} Katman"] = res
    print(f"{count:<15} | {res['rmse']:<10.5f} | {res['r2']:<10.5f} | {res['time']:<10.2f}")

plot_experiment(results_layers, "Gizli Katman", "2 Katman", "5_katman_sayisi_deneyi")


# =====================================================================
# ADIM 6: GİZLİ KATMANDAKİ NÖRON SAYISI DENEYİ
# =====================================================================
print("\n=== 6. GİZLİ KATMANDAKİ NÖRON SAYISI DENEYİ ===")
neuron_counts = [2, 5, 10, 20, 50]
results_neurons = {}

print(f"{'Nöron Sayısı':<15} | {'RMSE':<10} | {'R2':<10} | {'Süre (sn)':<10}")
print("-" * 55)

for num in neuron_counts:
    def build_neuron_model(n=num):
        return keras.Sequential([
            layers.Input(shape=(1,)),
            layers.Dense(n, activation='sigmoid', kernel_initializer=keras.initializers.RandomNormal(seed=SEED)),
            layers.Dense(1, activation='linear', kernel_initializer=keras.initializers.RandomNormal(seed=SEED))
        ])
    res = train_and_evaluate(build_neuron_model, optimizer=keras.optimizers.Adam(learning_rate=0.01))
    results_neurons[f"{num} Nöron"] = res
    print(f"{num:<15} | {res['rmse']:<10.5f} | {res['r2']:<10.5f} | {res['time']:<10.2f}")

plot_experiment(results_neurons, "Nöron Sayısı", "5 Nöron", "6_noron_sayisi_deneyi")


# =====================================================================
# ADIM 7: FİNAL / OPTİMUM MODEL DENEYİ
# =====================================================================
print("\n=== 7. FİNAL OPTİMUM MODEL EĞİTİMİ ===")
def build_final_model():
    return keras.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(5, activation='sigmoid', kernel_initializer=keras.initializers.RandomNormal(seed=SEED)),
        layers.Dense(5, activation='sigmoid', kernel_initializer=keras.initializers.RandomNormal(seed=SEED)),
        layers.Dense(1, activation='linear', kernel_initializer=keras.initializers.RandomNormal(seed=SEED))
    ])

final_res = train_and_evaluate(build_final_model, optimizer=keras.optimizers.Adam(learning_rate=0.01))
print(f"Final Model RMSE: {final_res['rmse']:.5f} | R2: {final_res['r2']:.5f} | Süre: {final_res['time']:.2f} sn")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(final_res["rmse_curve"], color='crimson', linewidth=2)
axes[0].set_yscale('log')
axes[0].set_title("Nihai Kayıp (Loss / RMSE)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (RMSE)")
axes[0].grid(True, which="both", ls="--", alpha=0.6)

axes[1].plot(p, y, label='Hedef', color='midnightblue', linewidth=2)
axes[1].plot(p, final_res["y_pred"], label='Optimum Ağ Çıkışı', color='orangered', linestyle='--', linewidth=2)
axes[1].set_title("Hedef vs Optimum Ağ Çıkışı")
axes[1].set_xlabel("Giriş (p)")
axes[1].set_ylabel("Çıkış (y)")
axes[1].legend()
axes[1].grid(True, ls="--", alpha=0.6)

axes[2].plot(range(1, len(final_res["r2_curve"]) + 1), final_res["r2_curve"], color='forestgreen', linewidth=2)
axes[2].set_title(f"Nihai R2 Skoru (Son: {final_res['r2']:.4f})")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("R2 Score")
axes[2].grid(True, ls="--", alpha=0.6)

plt.tight_layout()
plt.savefig("grafikler/7_final_optimum_model.png", dpi=200)
plt.show()

print("\nTüm deneyler tamamlandı! Grafikler 'grafikler/' klasörüne kaydedildi.")