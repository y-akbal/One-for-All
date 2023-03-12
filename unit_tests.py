
from abc import ABC, abstractmethod
 
class shape_test(ABC):
     
    @abstractmethod
    def test_shapes(self, model, input_):
        pass
    
    @abstractmethod
    def output_shapes(self):
        pass
    
    @abstractmethod
    def count_params(self, model):
    	pass



