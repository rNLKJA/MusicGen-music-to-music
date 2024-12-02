from app import app

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(port=8000, debug=True)