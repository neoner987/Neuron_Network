import main as m
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    firBais = np.zeros(m.secLayer_size)
    secBais = np.zeros(m.thrLayer_size)
    firWeight = m.rng.uniform(-m.random_range, m.random_range, (m.Input_size, m.secLayer_size))
    secWeight = m.rng.uniform(-m.random_range, m.random_range, (m.secLayer_size, m.thrLayer_size))
    np.save(Path("savedData/softMax/W1.npy"), firWeight)
    np.save(Path("savedData/softMax/W2.npy"), secWeight)
    np.save(Path("savedData/softMax/B1.npy"), firBais)
    np.save(Path("savedData/softMax/B2.npy"), secBais)