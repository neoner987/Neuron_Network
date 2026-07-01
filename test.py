import main as m
import numpy as np
from pathlib import Path
from PIL import Image
import random as r

_trys = 0
_correct_trys = 0
_text_path = "data/testing/"
_Input = np.zeros(m.Input_size)
Test_input_neuron = np.zeros(m.Input_size)

if __name__ == "__main__":
    Bais1 = np.load(Path("savedData/softMax/B1.npy"))
    Bais2 = np.load(Path("savedData/softMax/B2.npy"))
    Weight1 = np.load(Path("savedData/softMax/W1.npy"))
    Weight2 = np.load(Path("savedData/softMax/W2.npy"))
    while True:
        _test_num = r.randint(0, 9)
        _text_path += str(_test_num)
        _file_path = Path(_text_path + "/" + str(r.randint(0, 60000)) + ".png")
        if _file_path.is_file():
            _img = Image.open(_file_path).convert('L')
            _pixels = np.array(_img, dtype=np.float32).flatten()
            _input_neuron = m.normalize_pixel(_pixels)
            Layer2 = m.sigma(np.dot(_input_neuron, Weight1) - Bais1)
            Layer3 = m.softMax(np.dot(Layer2, Weight2) - Bais2)

            _max_arr = np.argmax(Layer3)
            if _max_arr == _test_num:
                _correct_trys += 1
            else:
                #print(file_path)
                pass
            _trys += 1
            print(_correct_trys / _trys * 100)
        _text_path = _text_path.replace(str(_test_num), "")