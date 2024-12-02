import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import os
from dotenv import load_dotenv

class MusicGenerator:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Configure device with proper fallback handling
        self.device = self._get_device()
        
        try:
            self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
            self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
            
            # Move model to device with error handling
            try:
                self.model = self.model.to(self.device)
            except RuntimeError as e:
                self.device = torch.device('cpu')
                self.model = self.model.to(self.device)
                
        except Exception as e:
            raise
    
    def _get_device(self):
        if torch.backends.mps.is_available():
            # Check if MPS fallback is enabled
            if os.getenv('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
                return torch.device('cpu')
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
        
    def generate(self, prompt):
        try:
            # Clear CUDA cache if using GPU
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )
            
            # Move inputs to device safely
            try:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            except RuntimeError as e:
                # Fallback to CPU if device transfer fails
                self.device = torch.device('cpu')
                self.model = self.model.to(self.device)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with memory optimization
            with torch.inference_mode():
                audio_values = self.model.generate(
                    **inputs,
                    do_sample=True,
                    guidance_scale=1,
                    max_new_tokens=1024
                )
            
            # Move output to CPU and convert to numpy
            audio_values = audio_values.cpu()
            
            return {
                'audio_values': audio_values[0, 0].numpy(),
                'sampling_rate': self.model.config.audio_encoder.sampling_rate
            }
            
        except Exception as e:
            raise
        finally:
            # Clean up memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

# Initialize a single instance to be used across requests
try:
    music_generator = MusicGenerator()
except Exception as e:
    raise