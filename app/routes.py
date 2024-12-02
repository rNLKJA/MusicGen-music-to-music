from flask import request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
from app import app
from app.musicgen import music_generator
from app.audio_analyzer import AudioAnalyzer
import scipy.io.wavfile
import io

# Initialize AudioAnalyzer
audio_analyzer = AudioAnalyzer()

# Configure upload settings
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

# Ensure upload directory exists with proper permissions
try:
    os.makedirs(UPLOAD_FOLDER, mode=0o777, exist_ok=True)
    print(f"Upload directory created/verified at: {UPLOAD_FOLDER}")
except Exception as e:
    print(f"Error creating upload directory: {e}")

@app.route('/generate', methods=['POST'])
def generate_music():
    try:
        # Get prompt from JSON request
        data = request.get_json()
        if not data or 'prompt' not in data:
            return {'error': 'No prompt provided'}, 400
        
        prompt = data['prompt']
        
        # Generate audio
        result = music_generator.generate(prompt)
        
        # Create a bytes buffer for the WAV file
        audio_buffer = io.BytesIO()
        scipy.io.wavfile.write(
            audio_buffer,
            rate=result['sampling_rate'],
            data=result['audio_values']
        )
        audio_buffer.seek(0)
        
        # Return the audio file
        return send_file(
            audio_buffer,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='generated_music.wav'
        )

    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/analyze', methods=['POST'])
def analyze_music():
    try:
        # Check if there's a file in the request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file uploaded'
            }), 400
            
        file = request.files['file']
        
        # Check if a file was selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            }), 400
            
        # Validate file extension
        if not file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
            return jsonify({
                'success': False,
                'message': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
            
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        try:
            file.save(filepath)
            print(f"✅ File saved successfully at: {filepath}")
            
            # Initialize audio analyzer if not already done
            analyzer = AudioAnalyzer()
            
            # Extract features
            features = analyzer.extract_features(filepath)
            
            # Generate description
            description = analyzer.generate_description_from_features(features)
            
            # Clean up - remove temporary file
            os.remove(filepath)
            
            # Return analysis results
            return jsonify({
                'success': True,
                'message': 'Analysis completed successfully',
                'analysis': {
                    'technical_features': features,
                    'description': description
                }
            })
            
        except Exception as e:
            # Clean up file if it exists
            if os.path.exists(filepath):
                os.remove(filepath)
            raise e
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error during analysis: {str(e)}'
        }), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        'success': True,
        'message': 'Server is running'
    })

@app.route('/analyze-and-generate', methods=['POST'])
def analyze_and_generate():
    try:
        # Check if there's a file in the request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file uploaded'
            }), 400
            
        file = request.files['file']
        
        # Check if a file was selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            }), 400
            
        # Validate file extension
        if not file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
            return jsonify({
                'success': False,
                'message': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
            
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        try:
            # Save and analyze original file
            file.save(filepath)
            print(f"✅ File saved successfully at: {filepath}")
            
            # Initialize audio analyzer
            analyzer = AudioAnalyzer()
            
            # Extract features and generate description
            features = analyzer.extract_features(filepath)
            description = analyzer.generate_description_from_features(features)
            
            # Generate prompt for music generation
            generation_prompt = f"""
            Create a new piece of music inspired by these characteristics:
            - Tempo around {int(features['tempo'])} BPM
            - Key of {features['key']}
            - Similar energy level and tonal qualities
            
            Additional context: {description}
            """
            
            # Generate new music based on the analysis
            print("Generating new music...")
            result = music_generator.generate(generation_prompt)
            
            # Create a bytes buffer for the WAV file
            audio_buffer = io.BytesIO()
            scipy.io.wavfile.write(
                audio_buffer,
                rate=result['sampling_rate'],
                data=result['audio_values']
            )
            audio_buffer.seek(0)
            
            # Clean up original file
            os.remove(filepath)
            
            # Return the generated audio file directly
            return send_file(
                audio_buffer,
                mimetype='audio/wav',
                as_attachment=True,
                download_name='generated_music.wav'
            )
            
        except Exception as e:
            # Clean up file if it exists
            if os.path.exists(filepath):
                os.remove(filepath)
            raise e
        
    except Exception as e:
        print(f"❌ Error during analysis and generation: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error during analysis and generation: {str(e)}'
        }), 500



# Error handlers
@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'success': False,
        'message': 'Internal server error occurred'
    }), 500

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'success': True,
        'message': 'Server is running'
    })