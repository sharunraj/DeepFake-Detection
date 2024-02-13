import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths to preprocessed data directories
real_data_dir = 'OP-Real'
fake_data_dir = 'OP-Fake'

# Define output directories for train, test, and validation sets
train_dir = 'Finalv3/train'
test_dir = 'Finalv3/test'
val_dir = 'Finalv3/validation'

# Create output directories if they don't exist
os.makedirs(os.path.join(train_dir, 'real'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'fake'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'real'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'fake'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'real'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'fake'), exist_ok=True)

# Get list of filenames for real and fake data
real_filenames = os.listdir(real_data_dir)
fake_filenames = os.listdir(fake_data_dir)

# Define train/test/validation split ratios
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1

# Split real data into train, test, and validation sets
real_train, real_test_val = train_test_split(
    real_filenames, test_size=(test_ratio + val_ratio), random_state=42)
real_test, real_val = train_test_split(
    real_test_val, test_size=val_ratio/(test_ratio + val_ratio), random_state=42)

# Split fake data into train, test, and validation sets
fake_train, fake_test_val = train_test_split(
    fake_filenames, test_size=(test_ratio + val_ratio), random_state=42)
fake_test, fake_val = train_test_split(
    fake_test_val, test_size=val_ratio/(test_ratio + val_ratio), random_state=42)

# Copy real data to train, test, and validation directories
for filename in real_train:
    shutil.copy(os.path.join(real_data_dir, filename),
                os.path.join(train_dir, 'real', filename))

for filename in real_test:
    shutil.copy(os.path.join(real_data_dir, filename),
                os.path.join(test_dir, 'real', filename))

for filename in real_val:
    shutil.copy(os.path.join(real_data_dir, filename),
                os.path.join(val_dir, 'real', filename))

# Copy fake data to train, test, and validation directories
for filename in fake_train:
    shutil.copy(os.path.join(fake_data_dir, filename),
                os.path.join(train_dir, 'fake', filename))

for filename in fake_test:
    shutil.copy(os.path.join(fake_data_dir, filename),
                os.path.join(test_dir, 'fake', filename))

for filename in fake_val:
    shutil.copy(os.path.join(fake_data_dir, filename),
                os.path.join(val_dir, 'fake', filename))

print("Train/Test/Validation split completed.")
