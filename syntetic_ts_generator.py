import numpy as np
from typing import Callable, Tuple, Any, List
## import dataclasses
from dataclasses import dataclass
import tqdm as tqdm
from statsmodels.tsa.arima_process import ArmaProcess


@dataclass
class parameters:
    ## -- ## 
    seed: int = 422
    ## -- ## 
    min_length: int = 100 
    max_length: int = 1000 
    ## -- ##
    sin_wave: bool = True
    square_wave: bool = True
    triangle_wave: bool = True
    sawtooth_wave: bool = True
    arma: bool = True
    composite_wave: bool = True
    ##
    shift: float = 1.0 
    max_ar_order: int = 4
    max_ma_order: int = 4
    noise: bool = True 
    max_noise_std: float = 1.0
    clip_value: float = 5
    normalize: bool = True
    augment: bool = True
    ## -- ##

class WaveType:
    def __init__(self, params: parameters):
        self.params = params
        self.available_wave_types: List[str] = []
        self.__prep__()
    
    def generate_wave(self, 
                      amp: float, 
                      freq: float, 
                      length: int, 
                      noise_std: float = 0.0,
                      phase: float = 0.0,
                      shift: float = 0.0,
                      ar_order:int = 2,
                      ma_order:int = 3,) -> np.ndarray:
        """
        Generate a wave of the specified type.
        """
        wave_type = np.random.choice(
            self.available_wave_types
        )
        print(f"Generating wave of type: {wave_type}")
        t = np.arange(length)
        if wave_type == "sin_wave":
            wave = amp * np.sin(2 * np.pi * freq * t + phase) + shift
        elif wave_type == "square_wave":    
            wave = amp * np.sign(np.sin(2 * np.pi * freq * t + phase)) + shift
        elif wave_type == "triangle_wave":
            wave = amp * (2 * np.abs(2 * (t * freq + phase) % 2 - 1) - 1) + shift
        elif wave_type == "sawtooth_wave":
            wave = amp * (2 * (t * freq + phase) % 1 - 1) + shift
        elif wave_type == "arma_process":
            return self.generate_arma_process(
                length=length, 
                amp=amp, 
                ar_order=ar_order, 
                ma_order=ma_order
            )
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, length)
            wave += noise
        return wave.astype(np.float32)
        
    
    
    def generate_arma_process(self, 
                              length: int,
                              amp: float = 1.0,
                              ar_order: int = 2,
                              ma_order: int = 2,
                              ) -> np.ndarray:
        # Placeholder for ARMA process generation logic
        ma_roots = self.__pick_roots__(order=ma_order, inside_unit=False if np.random.rand() < 0.5 else True)
        ar_roots = self.__pick_roots__(order=ar_order, inside_unit=False)
        process = ArmaProcess.from_roots(maroots = ma_roots,  
                                         arroots = ar_roots)
        sample = process.generate_sample(
            nsample=length, 
            scale=amp, 
            burnin=100, 
            distrvs=np.random.normal,
        )
        return sample.astype(np.float32)
   
    
    def generate_composite_wave(self, **kwargs) -> np.ndarray:
        wave_1 = self.generate_wave(**kwargs)
        wave_2 = self.generate_wave(**kwargs)
        return (wave_1 + wave_2).astype(np.float32)
    ## -- ##
    def __pick_roots__(self, order: int = 4, 
               inside_unit:bool = False) -> np.ndarray:
        ## Adjust the number of roots based on the order
        num_real_roots = np.random.randint(order + 1)
        num_complex_roots = order - num_real_roots
        low, high = (0, 1) if inside_unit else (1, 10.0)

        if num_complex_roots%2 == 1:
            num_complex_roots += 1

        real_roots: np.ndarray | List[None] = []
        complex_roots: np.ndarray | List[None] = []

        if num_complex_roots > 0:
            x,y = np.random.uniform(-1, 1, num_complex_roots // 2), np.random.uniform(-1, 1, num_complex_roots // 2)
            complex_roots = np.concatenate([
                x + 1j * y, 
                x - 1j * y
            ])
            complex_roots/= np.abs(complex_roots)
            scaler = np.random.uniform(low, high, size = num_complex_roots // 2)
            complex_roots *= np.concatenate([scaler, scaler])
            
        ## Generate real roots
        if num_real_roots > 0:  
            real_roots = np.random.uniform(low, high, size = num_real_roots)

        return np.concatenate([real_roots, complex_roots])

    def __prep__(self):
        """
        Placeholder for prediction logic.
        This method can be used to predict the next value in the time series.
        """
        if self.params.sin_wave:
            self.available_wave_types.append("sin_wave")
        if self.params.square_wave:
            self.available_wave_types.append("square_wave")
        if self.params.triangle_wave:
            self.available_wave_types.append("triangle_wave")
        if self.params.sawtooth_wave:
            self.available_wave_types.append("sawtooth_wave")
        if self.params.arma:
            self.available_wave_types.append("arma_process")
        if not self.available_wave_types:
            raise ValueError("No wave types available for generation. Please enable at least one wave type.")
    
    def __generate_wave_params(self):
        wave_params = {
            "freq": np.random.uniform(0.01, 10.0),
            "amplitude": np.random.uniform(0.1, 5.0),
            "phase": np.random.uniform(0, 2 * np.pi),
            "length": np.random.randint(self.params.min_length, self.params.max_length),
            "noise_std": np.random.uniform(0, self.params.max_noise_std) if self.params.noise else 0.0,
            "shift": self.params.shift if self.params.shift is not None else 0.0,
            "ar_order": None,
            "ma_order": None
        }
        if self.params.arma:
            # Generate ARMA parameters
            p = np.random.randint(0, self.params.max_ar_order + 1)
            q = np.random.randint(0, self.params.max_ma_order + 1)
            # For simplicity, we can use a fixed ARMA process generation function
            # In practice, you would generate AR and MA coefficients based on p and q
            wave_params["ar_order"], wave_params["ma_order"] = p, q            
        return wave_params


    def generate(self, 
                 length: int = None,
                 train: bool = True)-> np.ndarray:
        wave_params = self.__generate_wave_params()
        ## open up params
        freq, amp, phase, length_, noise_std, shift, ar_order, ma_order = (
            wave_params["freq"],
            wave_params["amplitude"],
            wave_params["phase"],
            wave_params["length"],
            wave_params["noise_std"],
            wave_params["shift"],
            wave_params["ar_order"],
            wave_params["ma_order"]
        )
        length = length if length is not None else length_
        if np.random.rand() < 0.5 and self.params.composite_wave:
            temp_data = self.generate_composite_wave(
                amp=amp, 
                freq=freq, 
                length=length, 
                noise_std=noise_std, 
                phase=phase, 
                shift=shift,
                ar_order=ar_order,
                ma_order=ma_order,
            )   
        else:
            temp_data = self.generate_wave(
                amp=amp, 
                freq=freq, 
                length=length, 
                noise_std=noise_std, 
                phase=phase, 
                shift=shift,
                ar_order=ar_order,
                ma_order=ma_order,
            )
        if self.params.clip_value is not None:
            temp_data = np.clip(temp_data, -self.params.clip_value, self.params.clip_value)
        if self.params.normalize:
            temp_data = (temp_data - np.mean(temp_data)) / np.std(temp_data)
        if train:
            return temp_data
        if np.random.rand() < 0.5 and self.params.augment:
            temp_data = self._augment_(temp_data)
            if np.random.rand() < 0.1:
                # Double augmentation
                temp_data = self._augment_(temp_data)
        return temp_data
    def _augment_(self, data: np.ndarray) -> np.ndarray:
        """
        Placeholder for augmentation logic.
        This method can be used to augment the time series data.
        """
        augmentations_fn = [
            lambda x: np.flip(x),  # Flip the time series
            lambda x: np.roll(x, shift=np.random.randint(1, 10)),  # Roll the time series
            lambda x: x + np.random.normal(0, 1, size=x.shape),  # Add noise
            lambda x: x * (1 + np.random.uniform(-0.5, 0.5)),  # Scale the time series, 
        ]
        augmentation_fn = np.random.choice(augmentations_fn)
        initial_pt = np.random.randint(0, len(data)*0.95)
        end_pt = np.random.randint(initial_pt, len(data))
        data[initial_pt:end_pt] = augmentation_fn(data[initial_pt:end_pt])
        return data

        
        
"""
params = parameters()
ts = WaveType(params)

from matplotlib import pyplot as plt
plt.plot(ts.generate(250))
plt.show()

"""
    