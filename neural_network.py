veriler = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]

w1 = 0.3
w2 = -0.2
bias = 0.1
learning_rate = 0.1

batch_size = 4
epoch_size = 4
epoch = 0
hata = 1

while hata != 0 and epoch < 10000:
    hata = 0
    epoch += 1

    for x1, x2, hedef in veriler:
        sonuc = w1 * x1 + w2 * x2 + bias
        tahmin = 1 if sonuc >= 0 else 0

        if tahmin != hedef:
            hata += 1
            w1 += learning_rate * (hedef - tahmin) * x1
            w2 += learning_rate * (hedef - tahmin) * x2
            bias += learning_rate * (hedef - tahmin)

    print("Epoch:", epoch, "Hata:", hata)

print("Egitim tamamlandi.")
print("Son agirliklar:", w1, w2, bias)
print("Batch size:", batch_size, "Epoch size:", epoch_size)
