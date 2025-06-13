# Install required dependencies


# Import libraries
import zipfile
import os
from PIL import Image
from tqdm.notebook import tqdm  # Use tqdm.notebook for Jupyter progress bars

def extract_zip(zip_path, extract_path):
    """Extracts a ZIP file to the specified path with progress feedback."""
    try:
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"ZIP file not found at: {zip_path}")
        
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            total = len(zip_ref.infolist())
            print(f"Extracting {total} files from {zip_path}...")
            for member in tqdm(zip_ref.infolist(), total=total, desc="Extracting"):
                # Avoid overwriting existing files
                target_path = os.path.join(extract_path, member.filename)
                if not os.path.exists(target_path):
                    zip_ref.extract(member, extract_path)
        print(f"✅ Extraction complete to {extract_path}")
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid ZIP file")
        raise
    except Exception as e:
        print(f"Error during extraction: {e}")
        raise

def process_and_crop_images(input_folder, output_folder, target_size=(150, 150)):
    """Processes images by center-cropping to square and resizing."""
    expected_categories = ['NORMAL', 'PNEUMONIA']
    
    # Validate input folder structure
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    for category in expected_categories:
        category_path = os.path.join(input_folder, category)
        save_path = os.path.join(output_folder, category)

        if not os.path.exists(category_path):
            print(f"Skipping: {category_path} (does not exist)")
            continue

        os.makedirs(save_path, exist_ok=True)
        image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing {len(image_files)} images in {category}...")
        for filename in tqdm(image_files, desc=f"Processing {category}"):
            try:
                img_path = os.path.join(category_path, filename)
                save_img_path = os.path.join(save_path, filename)
                
                # Skip if output file already exists
                if os.path.exists(save_img_path):
                    print(f"Skipping: {save_img_path} (already exists)")
                    continue
                
                img = Image.open(img_path).convert('RGB')

                # Center crop to square
                w, h = img.size
                min_dim = min(w, h)
                img = img.crop((
                    (w - min_dim) // 2,
                    (h - min_dim) // 2,
                    (w + min_dim) // 2,
                    (h + min_dim) // 2
                ))

                # Resize to target size
                img = img.resize(target_size)

                # Save the processed image
                img.save(save_img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    print("✅ Image preprocessing complete.")

# Set paths (modify these based on your system)
zip_path = r'C:\Users\niran\Downloads\Pneumonia X-Ray Dataset.zip'
extract_path = 'raw_data'
input_path = 'raw_data'
output_path = 'processed_data'

# Step 1: Extract ZIP
extract_zip(zip_path, extract_path)

# Step 2: Process images
process_and_crop_images(input_path, output_path)