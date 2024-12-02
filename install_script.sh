#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${YELLOW}[*] $1${NC}"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}[+] $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}[-] $1${NC}"
}

# Check OS type
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS installation
    print_status "Detected macOS system"
    print_status "Installing Python 3.9 using Homebrew..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        print_status "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install Python 3.9
    brew install python@3.9
    
    # Create virtual environment
    print_status "Creating virtual environment..."
    python3.9 -m venv venv
    source venv/bin/activate
    
elif [[ "$(uname)" == "Linux" ]]; then
    # Linux installation
    print_status "Detected Linux system"
    
    # Install Python 3.9 dependencies
    print_status "Installing Python dependencies..."
    sudo apt-get update
    sudo apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev
    
    # Download and install Python 3.9
    print_status "Downloading Python 3.9..."
    wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz
    tar -xf Python-3.9.0.tgz
    cd Python-3.9.0
    ./configure --enable-optimizations
    make -j $(nproc)
    sudo make altinstall
    
    # Create virtual environment
    print_status "Creating virtual environment..."
    python3.9 -m venv venv
    source venv/bin/activate
    
else
    print_error "Unsupported operating system"
    exit 1
fi

# Upgrade pip
print_status "Upgrading pip..."
python -m pip install --upgrade pip

# Install required packages for MusicGen
print_status "Installing required packages..."
pip install flask
pip install torch torchaudio
pip install git+https://github.com/huggingface/transformers.git

print_success "Installation completed successfully!"
print_status "To activate the virtual environment, run: source venv/bin/activate"
print_status "To run the Flask app, use: python -m flask run --host=0.0.0.0 --port=8080 --debug"