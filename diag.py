import torch, timm, json, sys
import torch.nn.functional as F

# 1. Inspect checkpoint
ckpt = torch.load('models/dermai_best.pt', map_location='cpu')
num_ckpt_classes = ckpt['classifier.bias'].shape[0]
print(f"CHECKPOINT num_classes: {num_ckpt_classes}")

# 2. Check model_meta.json
with open('models/model_meta.json') as f:
    meta = json.load(f)
print(f"META num_classes: {meta['num_classes']}")
print(f"META class_names: {meta['class_names']}")

# 3. Load model with correct class count from checkpoint
net = timm.create_model('efficientnet_b4', pretrained=False, num_classes=num_ckpt_classes)
net.load_state_dict(ckpt)
net.eval()
print("\nModel loaded OK")

# 4. Dummy inference
dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = net(dummy)
probs = F.softmax(out, dim=1).squeeze().tolist()
print(f"Output classes: {len(probs)}, sum={sum(probs):.4f}")

# 5. Import cnn_classifier and check state
sys.path.insert(0, '.')
from algorithms.cnn_classifier import classify_image, CLASS_NAMES, MODEL_LOADED, NUM_CLASSES
print(f"\n--- cnn_classifier state ---")
print(f"MODEL_LOADED: {MODEL_LOADED}")
print(f"CLASS_NAMES ({len(CLASS_NAMES)}): {CLASS_NAMES}")
print(f"NUM_CLASSES: {NUM_CLASSES}")
print(f"CLASS_NAMES count matches checkpoint: {len(CLASS_NAMES) == num_ckpt_classes}")

if len(CLASS_NAMES) != num_ckpt_classes:
    print(f"\nPROBLEM: model outputs {num_ckpt_classes} classes but CLASS_NAMES only has {len(CLASS_NAMES)}")
    print("This causes an IndexError during scores dict construction!")
else:
    print("\nClass count OK - scores mapping will work correctly")
