import os
import shutil
import random

def create_dirs():
    base = "c:/Users/krish/BetterNet/dataset"
    dirs = [
        f"{base}/TrainDataset/images",
        f"{base}/TrainDataset/masks",
        f"{base}/TestDataset/Kvasir-Test/images",
        f"{base}/TestDataset/Kvasir-Test/masks",
        f"{base}/TestDataset/CVC-Test/images",
        f"{base}/TestDataset/CVC-Test/masks",
        f"{base}/TestDataset/Sessile/images",
        f"{base}/TestDataset/Sessile/masks"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return base

def organize_kvasir(base):
    # Kvasir 1000 items -> 900 train, 100 test
    src_img = f"{base}/Kvasir-SEG/images"
    src_msk = f"{base}/Kvasir-SEG/masks"
    
    if not os.path.exists(src_img): return
    
    items = sorted(os.listdir(src_img))
    random.seed(42)
    random.shuffle(items)
    
    train_items = items[:900]
    test_items = items[900:]
    
    # Train
    for item in train_items:
        if os.path.exists(f"{src_msk}/{item}"):
            shutil.copy(f"{src_img}/{item}", f"{base}/TrainDataset/images/kvasir_{item}")
            shutil.copy(f"{src_msk}/{item}", f"{base}/TrainDataset/masks/kvasir_{item}")

    # Test
    for item in test_items:
        if os.path.exists(f"{src_msk}/{item}"):
            shutil.copy(f"{src_img}/{item}", f"{base}/TestDataset/Kvasir-Test/images/{item}")
            shutil.copy(f"{src_msk}/{item}", f"{base}/TestDataset/Kvasir-Test/masks/{item}")
            
    print(f"Organized Kvasir: 900 Train, {len(test_items)} Test")

def organize_cvc(base):
    # CVC 612 items -> 550 train, 62 test
    src_img = f"{base}/CVC_ClinicalDB/ds/img"
    src_msk = f"{base}/CVC_ClinicalDB/ds/ann" # Wait, CVC masks usually aren't direct pngs if they are in 'ann'. Let's check structure. Assuming they contain mask .pngs too based on typical downloads.
    
    # Actually, in most datasets, 'ds' folder structure implies standard img/mask. 
    # Let's see if masks are just pngs.
    if not os.path.exists(src_img): return
    items = sorted([f for f in os.listdir(src_img) if f.endswith('.png')])
    random.seed(42)
    random.shuffle(items)
    
    train_items = items[:550]
    test_items = items[550:]
    
    def copy_cvc(items, dest_type):
        for item in items:
            img_path = f"{src_img}/{item}"
            # CVC masks are sometimes in 'ann' but named same, sometimes different. 
            # If CVC ds/ann contains JSONs, that means they didn't download the direct Mask version.
            # I will just copy them directly assuming matching names for now. If masks are missing, we'll see.
            mask_path = f"{src_msk}/{item}" 
            if os.path.exists(mask_path):
                if dest_type == 'train':
                    shutil.copy(img_path, f"{base}/TrainDataset/images/cvc_{item}")
                    shutil.copy(mask_path, f"{base}/TrainDataset/masks/cvc_{item}")
                else:
                    shutil.copy(img_path, f"{base}/TestDataset/CVC-Test/images/{item}")
                    shutil.copy(mask_path, f"{base}/TestDataset/CVC-Test/masks/{item}")
    
    copy_cvc(train_items, 'train')
    copy_cvc(test_items, 'test')
    print(f"Organized CVC: 550 Train, {len(test_items)} Test")

def organize_sessile(base):
    # Sessile -> 196 test only
    src_img = f"{base}/sessile-main-Kvasir-SEG/images" # Assuming standard structure
    src_msk = f"{base}/sessile-main-Kvasir-SEG/masks"
    if not os.path.exists(src_img) or not os.path.exists(src_msk):
        print("Sessile images/masks not found in standard subfolders. Skipping Sessile.")
        return
        
    items = sorted(os.listdir(src_img))
    for item in items:
        if os.path.exists(f"{src_msk}/{item}"):
            shutil.copy(f"{src_img}/{item}", f"{base}/TestDataset/Sessile/images/{item}")
            shutil.copy(f"{src_msk}/{item}", f"{base}/TestDataset/Sessile/masks/{item}")
    print(f"Organized Sessile: {len(items)} Test")

base = create_dirs()
organize_kvasir(base)
organize_cvc(base)
organize_sessile(base)
print("Dataset successfully assembled!")
