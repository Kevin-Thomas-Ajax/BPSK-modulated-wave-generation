import numpy as np
import matplotlib.pyplot as plt

# Parameters
fc = 10  # Carrier frequency
fs = 1000  # Sampling frequency
T = 1 / fc  # Period of carrier signal
num_bits = 100  # Number of bits
bit_rate = 10  # Bit rate (bits per second)

# Generate binary data (random bits)
binary_data = np.random.randint(2, size=num_bits)

# Time array
num_samples = num_bits * int(fs/bit_rate)  # Adjusted to match the number of samples
t = np.arange(0, num_samples) / fs

# BPSK modulation
bpsk_signal = np.zeros(num_samples)

carrier = np.sin(2 * np.pi * fc * t)

for i, bit in enumerate(binary_data):
    if bit == 1:
        bpsk_signal[i*int(fs/bit_rate):(i+1)*int(fs/bit_rate)] = np.sin(2 * np.pi * fc * t[i*int(fs/bit_rate):(i+1)*int(fs/bit_rate)])
    else:
        bpsk_signal[i*int(fs/bit_rate):(i+1)*int(fs/bit_rate)] = -np.sin(2 * np.pi * fc * t[i*int(fs/bit_rate):(i+1)*int(fs/bit_rate)])

square_wave = np.repeat(binary_data, int(fs/bit_rate))

# Plot input binary data
plt.subplot(4, 1, 1)
plt.stem(np.arange(num_bits), binary_data, basefmt=' ')
plt.title('Input Binary Data')
plt.xlabel('Bit Index')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, square_wave, drawstyle='steps-post')
plt.title('Input Square Wave Data')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, carrier)
plt.title('BPSK Carrier Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot BPSK modulated signal
plt.subplot(4, 1, 4)
plt.plot(t, bpsk_signal)
plt.title('BPSK Modulated Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()
