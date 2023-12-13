import gdown
import zipfile

# Download the trained models
url = 'https://drive.google.com/uc?id=1gfsTX3lOYyAsxCzXSOAW9hR-ahGLEb3V'
output = '../trained_models.zip'
gdown.download(url, output, quiet=False)

# Unzip the downloaded file
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('../')