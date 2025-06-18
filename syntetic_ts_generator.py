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
    file_name: str = "synthetic_ts.dat" 
    order_name: str = "synthetic_ts_ord.dat" 
    temp_dir: str = "tmp" 
    ## -- ## 
    num_samples: int = 1000_000_000 
    min_length: int = 100 
    max_length: int = 1000 
    ## -- ##
    sin_wave: bool = True
    square_wave: bool = True 
    triangle_wave: bool = True
    sawtooth_wave: bool = True
    arma: bool = False
    composite_wave: bool = False
    ##
    shift: float = 0.0 
    max_ar_order: int = 4
    max_ma_order: int = 4
    noise: bool = True 
    max_noise_std: float = 0.1 
    clip_value: float = .5 
    normalize: bool = True
    ## -- ##

class WaveType:
    def __init__(self, params: parameters):
        self.params = params
        self.__prep__()
        self.available_wave_types: List[str] = []
    
    def generate_wave(self, 
                      amp: float, 
                      freq: float, 
                      length: int, 
                      noise_std: float = 0.0,
                      phase: float = 0.0,
                      shift: float = 0.0) -> np.ndarray:
        """
        Generate a wave of the specified type.
        """
        wave_type = np.choice([
            self.available_wave_types
        ])
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
                              noise_std: float = 0.0,
                              ) -> np.ndarray:
        # Placeholder for ARMA process generation logic
        ma_roots = self.__pick_roots__(order=ma_order, inside_unit=False)
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
        composite_wave = wave_1 + wave_2
        return composite_wave.astype(np.float32)
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
        available_wave_types = []
        if self.params.sin_wave:
            available_wave_types.append("sin_wave")
        if self.params.square_wave:
            available_wave_types.append("square_wave")
        if self.params.triangle_wave:
            available_wave_types.append("triangle_wave")
        if self.params.sawtooth_wave:
            available_wave_types.append("sawtooth_wave")
        if self.params.arma:
            available_wave_types.append("arma_process")
        if not available_wave_types:
            raise ValueError("No wave types available for generation. Please enable at least one wave type.")
        self.available_wave_types = available_wave_types
        
    
    def __generate_wave_params(self):
        wave_params = {
            "freq": np.random.uniform(0.1, 10.0),
            "amplitude": np.random.uniform(0.1, 5.0),
            "phase": np.random.uniform(0, 2 * np.pi),
            "length": np.random.randint(self.params.min_length, self.params.max_length),
            "noise_std": np.random.uniform(0, self.params.max_noise_std) if self.params.noise else 0.0,
            "shift": self.params.shift if self.params.shift is not None else 0.0
        }
        if self.params.arma:
            # Generate ARMA parameters
            p = np.random.randint(0, self.params.max_ar_order + 1)
            q = np.random.randint(0, self.params.max_ma_order + 1)
            # For simplicity, we can use a fixed ARMA process generation function
            # In practice, you would generate AR and MA coefficients based on p and q
            wave_params["arma_params"] = {"p": p, "q": q}
        return wave_params


    def generate(self)-> np.ndarray:
        wave_params = self.__generate_wave_params()
        ## open up params
        freq, amp, phase, length, noise_std, shift = (
            wave_params["freq"],
            wave_params["amplitude"],
            wave_params["phase"],
            wave_params["length"],
            wave_params["noise_std"],
            wave_params["shift"]
        )
        if np.random.rand() < 0.5 and self.params.composite_wave:
            temp_data = self.generate_composite_wave(
                amp=amp, 
                freq=freq, 
                length=length, 
                noise_std=noise_std, 
                phase=phase, 
                shift=shift
            )   
        else:
            temp_data = self.generate_wave(
                amp=amp, 
                freq=freq, 
                length=length, 
                noise_std=noise_std, 
                phase=phase, 
                shift=shift
            )
        if self.params.clip_value is not None:
            temp_data = np.clip(temp_data, -self.params.clip_value, self.params.clip_value)
        
        if self.params.normalize:
            temp_data = (temp_data - np.mean(temp_data)) / np.std(temp_data)
        return temp_data
        

def main():
    params = parameters()
    np.random.seed(params.seed)
    print(f"Generating synthetic time series with parameters: {params}")
    synt_data_generator = WaveType(params)
    counter = 0
    memmap_data = np.memmap(
        params.file_name, dtype=np.float32, mode='w+', shape=(params.num_samples, )
    )
    data_lengths = []
    while counter < params.num_samples:
        temp_data = synt_data_generator.generate()
        print(temp_data)
        memmap_data[counter:counter + len(temp_data)] = temp_data
        data_lengths.append(len(temp_data))       
        memmap_data.flush()
        counter += len(temp_data)
        #print(f"Generated {counter} samples", end='\r')
    data_lengths_memmap = np.memmap(
        params.order_name, dtype=np.int32, mode='w+', shape=(params.num_samples,)
    )
    data_lengths_memmap[:len(data_lengths)] = np.array(data_lengths, dtype=np.int32)
    data_lengths_memmap.flush()
    print(f"\nGenerated {counter} samples in total.")
    print(f"Data saved to {params.file_name} and lengths saved to {params.order_name}.")
    print("Done.")
        

    

    






if __name__ == "__main__":
    # oki doki, let's run the main function
    main()


