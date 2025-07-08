import os
import soundfile as sf
import numpy as np
import librosa
from tqdm import tqdm
from typing import List, Tuple
from natsort import natsorted

# ==========================#
#       I / O Functions     #
# ==========================#

def load_audio_files_from_dir(directory: str) -> List[Tuple[str, np.ndarray, int]]:
    """
    Example usage:
    
        audio_files = load_audio_files_from_dir(dir+cls)
        fn, audio, sr = audio_files[0]
    """
    audio_ext = ('.wav', '.mp3', '.flac', 'aiff', 'aif', '.ogg')
    audio_files = []
    for filename in tqdm(natsorted(os.listdir(directory)), desc="Loading audio files"):
        if filename.lower().endswith(audio_ext):
            file_path = os.path.join(directory, filename)
            audio, sr = librosa.load(file_path, sr=44100, mono=False)
            
            audio_files.append((filename, audio, sr))
        else:
            print(f"Skipping non-audio file: {filename}")
    return audio_files

def save_audio(cls_folder, save_dir, fn, y, sr):
    """
    Example usage:
    
        for i in range(len(audio_files)):
            filename, audio, sr = audio_files[i]
            print(f"Saving processed file {i+1}/{len(audio_files)}: {filename}")
            filename = f"c{cls_idx}_{i+1}.wav"
            save_audio(cls, filename, audio, sr)
    """
    # Save processed audio
    output_dir_16bit = save_dir + cls_folder
    if not os.path.exists(output_dir_16bit):
        os.makedirs(output_dir_16bit)
    output_path_16bit = os.path.join(output_dir_16bit, fn)
    # save as wav file 
    sf.write(output_path_16bit, y.T, sr, subtype='PCM_16', format='WAV')

def probe_format(fn, y, sr):
    """
    Example usage:
    
        for i in range(len(audio_files)):
            filename, audio, sr = audio_files[i]
            print(f"Probe format of file {i+1}/{len(audio_files)}: {filename}")
            probe_format(filename, audio, sr)
    """
    # Probe format details
    print(f"Sample rate: {sr}")
    print(f"Shape: {y.shape}")               # (n,) for mono, (channels, n) for stereo
    print(f"Dtype: {y.dtype}")               # Should be float32
    print(f"Value range: {y.min()} to {y.max()}")
    print('------')

def assert_format(fn, y, sr):
    """
    Example usage:
    
        for i in range(len(audio_files)):
            filename, audio, sr = audio_files[i]
            print(f"Assert format of file {i+1}/{len(audio_files)}: {filename}")
            assert_format(filename, audio, sr)
    """
    # Check format details
    assert sr == 44100, f"Sample rate mismatch for {fn}: expected 44100, got {sr}"
    assert y.shape == (2, 441000), f"Shape mismatch for {fn}: expected (2, 44100) or (44100,), got {y.shape}"
    assert y.dtype == np.float32, f"Dtype mismatch for {fn}: expected float32, got {y.dtype}"
    assert y.min() >= -1.0 and y.max() <= 1.0, f"Value range mismatch for {fn}: expected [-1.0, 1.0], got {y.min()} to {y.max()}"
    

# ===========================#
#       Sub-Processing       #
# ===========================#

def normalize_per_channel(y):
    """
    Normalize audio data per channel to the range [-1.0, 1.0].
    
    Args:
        y (np.ndarray): Audio data with shape (channels, samples).
        
    Returns:
        np.ndarray: Normalized audio data.
    """
    assert len(y.shape) == 2, f"Audio data should be 2D, got {y.shape}"
    
    # Avoid division by zero with small epsilon
    epsilon = 1e-8
    
    # Compute per-channel max absolute value
    max_vals = np.max(np.abs(y), axis=1, keepdims=True) + epsilon
    
    # Normalize each channel
    y = y / max_vals
    
    return y

def stereo(y):
    if len(y.shape) == 1:
        y = np.stack([y, y], axis=0)  # Convert mono to stereo
    return y

# ====================================#
# Class-specific Processing Functions #
# ====================================#

# c08_Bell
def process_audio_bell(fn, y, sr):
    
    # convert mono to stereo if needed
    if len(y.shape) == 1:
        y = np.stack([y, y], axis=0)  
    
    # take the first 3s then tile
    cur_len = y.shape[1]
    if cur_len < 441000:
        target_len = int(10.0 * sr)
        short_len = int(3.0 * sr)
        
        if cur_len < short_len:
            # pad with zeros if the audio is shorter than 3 seconds
            pad_len = short_len - cur_len
            y = np.pad(y, ((0, 0), (0, pad_len)), mode='constant')
            assert y.shape[1] == short_len, f"Audio length after padding is {y.shape[1]}, not {short_len} for {fn}"
            
        first_3s = y[:, :short_len]
        n_tiles = int(np.ceil(target_len / short_len))
        y = np.tile(first_3s, (1, n_tiles))[:, :target_len] 
    elif y.shape[1] >= 441000:
        y = y[:, :441000]
    
    y = normalize_per_channel(y)
    
    return fn, y, sr

# c09_Gunshot
def process_audio_gunshot(fn, y, sr):
    
    # convert mono to stereo if needed
    if len(y.shape) == 1:
        y = np.stack([y, y], axis=0)  
    
    cur_len = y.shape[1]
    
    # just tile; if shorter than 3, pad to 3s
    if cur_len < 441000:
        target_len = int(10.0 * sr)
        short_len = int(3.0 * sr)
        
        if cur_len < short_len:
            # pad with zeros if the audio is shorter than 3 seconds
            pad_len = short_len - cur_len
            y = np.pad(y, ((0, 0), (0, pad_len)), mode='constant')
            assert y.shape[1] == short_len, f"Audio length after padding is {y.shape[1]}, not {short_len} for {fn}"
            
        n_tiles = int(np.ceil(target_len / y.shape[1]))
        y = np.tile(y, (1, n_tiles))[:, :target_len] 
        
    elif y.shape[1] >= 441000:
        y = y[:, :441000]
    
    y = normalize_per_channel(y)
   
    return fn, y, sr

# c10_Keyboard
def process_audio_keyboard(fn, y, sr):

    # convert mono to stereo if needed
    if len(y.shape) == 1:
        y = np.stack([y, y], axis=0)  
    
    # just clip to 10s, there is no <10s audio in this class
    if y.shape[1] > 441000:
        y = y[:, :441000]
    
    # normalize per channel
    y = normalize_per_channel(y)
    
    return fn, y, sr

# c11_Alarm
def process_audio_alarm(fn, y, sr):
    
    # convert mono to stereo if needed
    if len(y.shape) == 1:
        y = np.stack([y, y], axis=0)  
    
    cur_len = y.shape[1]
    # just tile
    if cur_len < 441000:
        target_len = int(10.0 * sr)
        short_len = int(3.0 * sr)
            
        n_tiles = int(np.ceil(target_len / y.shape[1]))
        y = np.tile(y, (1, n_tiles))[:, :target_len] 
        
    elif y.shape[1] >= 441000:
        y = y[:, :441000]
    
    y = normalize_per_channel(y)
        
    assert y.shape[0] == 2, f"Audio is {y.shape}, not stereo for {fn}"
    assert y.shape[1] == 441000, f"Audio length is {y.shape}, not 441000 for {fn}"
    assert y.dtype == np.float32, f"Audio data type is {y.dtype}, not float32 for {fn}"
    
    return fn, y, sr

# c12_Sea
def process_audio_sea(fn, y, sr):
    
    # convert mono to stereo if needed
    if len(y.shape) == 1:
        y = np.stack([y, y], axis=0)  
    
    cur_len = y.shape[1]
    
    # just tile; if shorter than 5s, pad to 5s
    if cur_len < 441000:
        target_len = int(10.0 * sr)
        short_len = int(5.0 * sr)
        
        if cur_len < short_len:
            # pad with zeros if the audio is shorter than 3 seconds
            pad_len = short_len - cur_len
            y = np.pad(y, ((0, 0), (0, pad_len)), mode='constant')
            assert y.shape[1] == short_len, f"Audio length after padding is {y.shape[1]}, not {short_len} for {fn}"
            
        n_tiles = int(np.ceil(target_len / y.shape[1]))
        y = np.tile(y, (1, n_tiles))[:, :target_len] 
        
    elif y.shape[1] >= 441000:
        y = y[:, :441000]
    
    y = normalize_per_channel(y)
        
    assert y.shape[0] == 2, f"Audio is {y.shape}, not stereo for {fn}"
    assert y.shape[1] == 441000, f"Audio length is {y.shape}, not 441000 for {fn}"
    assert y.dtype == np.float32, f"Audio data type is {y.dtype}, not float32 for {fn}"

    return fn, y, sr

# c13_Rain
def process_audio_rain(fn, y, sr):
    
    # convert mono to stereo if needed
    y = stereo(y)
    
    cur_len = y.shape[1]
    # just tile
    if cur_len < 441000:
        target_len = int(10.0 * sr)
        short_len = int(3.0 * sr)
            
        n_tiles = int(np.ceil(target_len / y.shape[1]))
        y = np.tile(y, (1, n_tiles))[:, :target_len] 
        
    elif y.shape[1] >= 441000:
        y = y[:, :441000]
    
    y = normalize_per_channel(y)
        
    assert y.shape[0] == 2, f"Audio is {y.shape}, not stereo for {fn}"
    assert y.shape[1] == 441000, f"Audio length is {y.shape}, not 441000 for {fn}"
    assert y.dtype == np.float32, f"Audio data type is {y.dtype}, not float32 for {fn}"
    
    return fn, y, sr

# c14_Liquid
def process_audio_liquid(fn, y, sr):
    
    # convert mono to stereo if needed
    y = stereo(y) 
    
    cur_len = y.shape[1]
    
    # just tile; if shorter than 1.5s, pad to 1.5s
    if cur_len < 441000:
        target_len = int(10.0 * sr)
        short_len = int(1.5 * sr)
        
        if cur_len < short_len:
            # pad with zeros if the audio is shorter than 3 seconds
            pad_len = short_len - cur_len
            y = np.pad(y, ((0, 0), (0, pad_len)), mode='constant')
            assert y.shape[1] == short_len, f"Audio length after padding is {y.shape[1]}, not {short_len} for {fn}"
            
        n_tiles = int(np.ceil(target_len / y.shape[1]))
        y = np.tile(y, (1, n_tiles))[:, :target_len] 
        
    elif y.shape[1] >= 441000:
        y = y[:, :441000]
    
    y = normalize_per_channel(y)
        
    assert y.shape[0] == 2, f"Audio is {y.shape}, not stereo for {fn}"
    assert y.shape[1] == 441000, f"Audio length is {y.shape}, not 441000 for {fn}"
    assert y.dtype == np.float32, f"Audio data type is {y.dtype}, not float32 for {fn}"
    
    return fn, y, sr
