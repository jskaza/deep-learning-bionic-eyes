import pandas as pd
import zipfile
from PIL import Image
import io

df = pd.read_csv("../data/emotion/metadata.csv")

print(df[(df["implant_type"] == "12_20") & (df["emotion"] == "happy") & (df["data_filename"] == "9991.jpg")])



# Open the zip file and extract the specific image
with zipfile.ZipFile("../data/emotion/percepts.zip", "r") as zip_file:
    # Read the image file from the zip
    image_data = zip_file.read("emotionUpdated/3987.tif")
    # Create an image object from the bytes
    image = Image.open(io.BytesIO(image_data))
    # Display basic image information
    print("\nImage details:")
    print(f"Size: {image.size}")
    print(f"Mode: {image.mode}")
    # You can display the image using image.show() if needed
    image.show()

