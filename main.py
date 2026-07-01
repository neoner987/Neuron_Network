import numpy as np
import random as r
from PIL import Image
from pathlib import Path
import sys


Input_size= 784
secLayer_size = 100
thrLayer_size = 10
batch_size = 1000
text_path = "/data/training/"
activation_signal = 0.7
sigma_scale = 1
rng = np.random.default_rng()
random_range = 1
learning_rate = 1.05
trys = 0
correct_trys = 0
softT = 0.8


pixels = np.array([])
pixels.resize(Input_size)
input_neuron = np.array([])
input_neuron.resize(Input_size)


secLayer = np.zeros(secLayer_size)
thrLayer = np.zeros(thrLayer_size)

raw_secLayer = np.zeros(secLayer_size)
raw_thrLayer = np.zeros(thrLayer_size)

firWeight = np.array([])
secWeight = np.array([])
firWeight.resize(Input_size, secLayer_size)
secWeight.resize(secLayer_size, thrLayer_size)

firBais = np.array([])
secBais = np.array([])
firBais.resize(secLayer_size)
secBais.resize(thrLayer_size)

cost_func = 0

y = np.zeros(thrLayer_size)

grad_cost_b_3 = np.zeros(np.shape(secBais))
grad_cost_b_2 = np.zeros(np.shape(firBais))

grad_cost_w_3 = rng.uniform(np.shape(secWeight))
grad_cost_w_2 = rng.uniform(np.shape(firWeight))

def sigma(x):
    scaled_x = x / sigma_scale
    clipped_x = np.clip(scaled_x, -50, 50)
    return 1.0 / (1.0 + np.exp(-clipped_x))

def sigma_prime(x):
    scaled_x = x / sigma_scale
    clipped_x = np.clip(scaled_x, -50, 50)
    exp_neg = np.exp(-clipped_x)
    return exp_neg / ((1.0 + exp_neg) ** 2 * sigma_scale)

def normalize_pixel(pixel_array):
    return np.sin((pixel_array * np.pi) / 510) ** 2


def softMax(x):

    if softT == 0:
        ans = np.zeros(x.shape)
        ans[x == max(x)] = 1
        ans[x != max(x)] = 0
        return ans
    scaled_x = x / softT
    clipped_x = np.clip(scaled_x, -50, 50)
    exp_arr = np.exp(clipped_x)
    return exp_arr / exp_arr.sum()



if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    try:
        firBais = np.load(Path("savedData/softMax/B1.npy"))
        secBais = np.load(Path("savedData/softMax/B2.npy"))
        firWeight = np.load(Path("savedData/softMax/W1.npy"))
        secWeight = np.load(Path("savedData/softMax/W2.npy"))
    except:
        firBais = rng.uniform(-random_range, random_range, secLayer_size)
        secBais = rng.uniform(-random_range, random_range, thrLayer_size)
        firWeight = np.zeros((Input_size, secLayer_size))
        secWeight = np.zeros((secLayer_size, thrLayer_size))

    batchIndex = 0
    while True:
        test_num = r.randint(0, 9)
        text_path += str(test_num)
        file_path = Path(text_path + "/" + str(r.randint(0, 60000)) + ".png")
        if file_path.is_file():
            y[test_num] = 1
            img = Image.open(file_path).convert('L')
            pixels = np.array(img, dtype=np.float32).flatten()
            input_neuron = normalize_pixel(pixels)
            raw_secLayer = np.dot(input_neuron, firWeight) - firBais
            secLayer = sigma(raw_secLayer)
            raw_thrLayer = np.dot(secLayer, secWeight) - secBais
            thrLayer = softMax(raw_thrLayer)

            thrLayer_clipped = np.clip(thrLayer, 1e-15, 1.0 - 1e-15)
            cost_func -= np.sum(y * np.log(thrLayer_clipped))

            max_arr = np.argmax(thrLayer)

            deltab3 = thrLayer - y
            deltab2 = sigma_prime(raw_secLayer) * np.dot(secWeight, deltab3)
            deltaw3 = np.outer(secLayer, deltab3)
            deltaw2 = np.outer(input_neuron, deltab2)

            grad_cost_b_3 += deltab3
            grad_cost_b_2 += deltab2
            grad_cost_w_3 += deltaw3
            grad_cost_w_2 += deltaw2

            if max_arr == test_num:
                correct_trys += 1
            else:
                #print(file_path)
                pass
            trys += 1

            #print(str(test_num) + " | " + str(np.where(thrLayer == max(thrLayer))[0]))

            if batchIndex == batch_size:
                firBais -= (learning_rate * grad_cost_b_2) / batch_size
                secBais -= (learning_rate * grad_cost_b_3) / batch_size
                firWeight -= (learning_rate * grad_cost_w_2) / batch_size
                secWeight -= (learning_rate * grad_cost_w_3) / batch_size

                np.save(Path("savedData/softMax/W1.npy"), firWeight)
                np.save(Path("savedData/softMax/W2.npy"), secWeight)
                np.save(Path("savedData/softMax/B1.npy"), firBais)
                np.save(Path("savedData/softMax/B2.npy"), secBais)

                batchIndex = 0
                cost_func = 0

                grad_cost_b_3.fill(0)
                grad_cost_b_2.fill(0)
                grad_cost_w_3.fill(0)
                grad_cost_w_2.fill(0)
                print(correct_trys / trys * 100)
            batchIndex += 1
            y.fill(0)
        text_path = text_path.replace(str(test_num), "")

