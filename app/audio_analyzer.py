import torch
import torchaudio
import librosa
import numpy as np
from transformers import AutoProcessor, AutoModelForAudioClassification
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class AudioAnalyzer:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get Hugging Face token from environment
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        
        # Load genre classification model - using a public model instead
        try:
            self.genre_processor = AutoProcessor.from_pretrained(
                "mit/ast-finetuned-audioset-10-10-0.4593",
                token=self.hf_token
            )
            self.genre_model = AutoModelForAudioClassification.from_pretrained(
                "mit/ast-finetuned-audioset-10-10-0.4593",
                token=self.hf_token
            )
            self.genre_model.to(self.device)
        except Exception as e:
            print(f"Warning: Could not load genre classification model: {e}")
            self.genre_processor = None
            self.genre_model = None

    def analyze_audio(self, audio_path):
        """
        Analyze audio file and return text description
        """
        try:
            print(f"Starting analysis of file: {audio_path}")
            
            # Check if file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found at path: {audio_path}")
            
            # Get technical analysis
            print("Performing technical analysis...")
            technical_description = self._analyze_technical_features(audio_path)
            
            # Get GPT interpretation
            print("Getting GPT analysis...")
            creative_description = self._get_gpt_analysis(technical_description)
            
            print("Analysis completed successfully")
            return {
                'technical_analysis': technical_description,
                'creative_analysis': creative_description
            }
            
        except Exception as e:
            print(f"Error in analyze_audio: {str(e)}")
            raise Exception(f"Error analyzing audio: {str(e)}")

    def _analyze_technical_features(self, audio_path):
        """
        Analyze technical features of the audio
        """
        try:
            # Load and resample audio properly
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Ensure audio is the right shape (add channel dimension if needed)
            if len(y.shape) == 1:
                y = y.reshape(1, -1)
            
            # Convert to torch tensor
            waveform = torch.FloatTensor(y)
            
            # Extract features using librosa
            # Get tempo
            tempo, _ = librosa.beat.beat_track(y=y[0], sr=sr)
            
            # Get key
            key = librosa.feature.chroma_stft(y=y[0], sr=sr)
            
            # Get spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y[0], sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y[0], sr=sr)
            
            # Only attempt genre classification if model loaded successfully
            genre_info = ""
            if self.genre_processor is not None and self.genre_model is not None:
                try:
                    # Prepare inputs for genre classification
                    inputs = self.genre_processor(waveform, sampling_rate=sr, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get genre predictions
                    with torch.no_grad():
                        outputs = self.genre_model(**inputs)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    genre_info = f"\nGenre classification confidence: {predictions.max().item():.2f}"
                except Exception as e:
                    print(f"Warning: Genre classification failed: {e}")
                    genre_info = "\nGenre classification unavailable"
            
            # Generate technical description
            description = self._generate_description(
                tempo=tempo,
                key_features=key,
                spectral_centroids=spectral_centroids,
                spectral_rolloff=spectral_rolloff
            )
            
            return description + genre_info

        except Exception as e:
            print(f"Error in technical analysis: {str(e)}")
            raise Exception(f"Failed to analyze audio features: {str(e)}")

    def _get_gpt_analysis(self, technical_description):
        """
        Get creative analysis from GPT based on technical features
        """
        try:
            prompt = f"""
            As a music expert, analyze this technical description of a piece of music and provide 
            an insightful, creative interpretation that explains its musical qualities, emotional 
            character, and overall feel in natural language. Focus on how the different elements 
            work together to create the overall musical experience.

            Technical Description:
            {technical_description}
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4",  # or gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": "You are a knowledgeable music expert providing insightful analysis of musical pieces."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error getting creative analysis: {str(e)}"

    def _generate_description(self, tempo, key_features, spectral_centroids, spectral_rolloff):
        """
        Generate technical description from audio features
        """
        try:
            # Get dominant key
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_profile = np.mean(key_features, axis=1)
            dominant_key = key_names[np.argmax(key_profile)]
            
            # Analyze brightness/timbre
            avg_brightness = np.mean(spectral_centroids)
            brightness = "bright" if avg_brightness > 2000 else "warm"
            
            # Format description
            description = f"""
            Technical Analysis:
            - Tempo: {int(tempo)} BPM
            - Key: {dominant_key}
            - Timbre: The song has a {brightness} tonal quality
            - Overall character: {self._describe_spectral_features(spectral_rolloff)}
            """
            
            return description.strip()
            
        except Exception as e:
            print(f"Error generating description: {str(e)}")
            return "Could not generate technical description"
    
    def _describe_spectral_features(self, spectral_rolloff):
        """
        Generate description of spectral features
        """
        avg_rolloff = np.mean(spectral_rolloff)
        if avg_rolloff > 3000:
            return "prominent high frequencies, suggesting bright instrumentation"
        elif avg_rolloff > 1500:
            return "balanced frequency distribution with mixed instrumentation"
        else:
            return "emphasis on lower frequencies, suggesting bass-heavy elements"

    def extract_features(self, audio_path):
        """
        Extract technical features from audio file and return as structured data
        """
        try:
            # Load and resample audio
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Ensure audio is the right shape
            if len(y.shape) == 1:
                y = y.reshape(1, -1)
            
            # Extract basic features
            tempo, _ = librosa.beat.beat_track(y=y[0], sr=sr)
            
            # Extract key
            chroma = librosa.feature.chroma_stft(y=y[0], sr=sr)
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_profile = np.mean(chroma, axis=1)
            dominant_key = key_names[np.argmax(key_profile)]
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y[0], sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y[0], sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y[0], sr=sr)[0]
            
            # Calculate average energy in different frequency bands
            spec = np.abs(librosa.stft(y[0]))
            
            # Return structured data
            return {
                'tempo': float(tempo),
                'key': dominant_key,
                'spectral_analysis': {
                    'brightness': float(np.mean(spectral_centroids)),
                    'rolloff': float(np.mean(spectral_rolloff)),
                    'bandwidth': float(np.mean(spectral_bandwidth)),
                    'spectral_stats': {
                        'mean': float(np.mean(spec)),
                        'std': float(np.std(spec)),
                        'max': float(np.max(spec))
                    }
                },
                'audio_quality': {
                    'sample_rate': sr,
                    'duration': float(len(y[0]) / sr),
                    'rms_energy': float(np.sqrt(np.mean(y[0]**2)))
                }
            }
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            raise Exception(f"Failed to extract audio features: {str(e)}")

    def generate_description_from_features(self, features):
        """
        Generate a natural language description of the music using OpenAI based on extracted features
        """
        try:
            # Format features into a readable string
            feature_text = f"""
            Musical Features:
            - Tempo: {features['tempo']} BPM
            - Musical Key: {features['key']}
            - Brightness: {features['spectral_analysis']['brightness']:.2f}
            - Frequency Distribution: {features['spectral_analysis']['rolloff']:.2f}
            - Duration: {features['audio_quality']['duration']:.2f} seconds
            - Energy Level: {features['audio_quality']['rms_energy']:.4f}
            """
            
            prompt = f"""
            As a music expert, provide a brief, engaging description of this piece of music based on 
            its technical characteristics. Focus on how these elements combine to create the overall 
            musical experience. Keep the description concise (2-3 sentences).

            {feature_text}
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable music expert providing concise, insightful analysis of musical pieces."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating description: {str(e)}")
            return f"Error generating description: {str(e)}"