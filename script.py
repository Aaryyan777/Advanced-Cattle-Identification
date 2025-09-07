# ================================================================================================
# Single-Model SOTA: EfficientNetV2-S for Indian Bovine Breed Classification (Fast 85%+)
# - Uses tf.data for stable labels and fast I/O
# - Correct EfficientNetV2 preprocessing
# - Class balancing + label smoothing + cosine LR
# - Strong but safe augmentations
# ================================================================================================

# 0) Optional: pin TF for reproducibility (comment out if Colab already has recent TF)
# !pip install -q tensorflow==2.12.0

import os, math, pathlib, numpy as np, tensorflow as tf
from sklearn.utils import class_weight

print(tf.__version__)
AUTOTUNE = tf.data.AUTOTUNE
SEED = 1337

# 1) Paths and dataset download (Kaggle)
PROJECT_DIR = '/content/efnv2s_bovine'
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Upload kaggle.json if not present
if not os.path.exists("/root/.kaggle/kaggle.json"):
    print("Please upload kaggle.json")
    from google.colab import files
    uploaded = files.upload()
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json

# Download and unzip the dataset
!kaggle datasets download -d lukex9442/indian-bovine-breeds -p {DATA_DIR} --unzip

# 2) Find the directory that contains class subfolders
def find_image_dir(base):
    for root, dirs, files in os.walk(base):
        dirs_alpha = [d for d in dirs if any(c.isalpha() for c in d)]
        if len(dirs_alpha) >= 10:  # dataset has many breed folders
            return root
    raise FileNotFoundError("Could not find a folder with multiple class subfolders.")
IMAGE_DIR = find_image_dir(DATA_DIR)
print("Using image directory:", IMAGE_DIR)

# 3) Configs
IMG_SIZE = 300        # 300x300 is a sweet spot for EfficientNetV2-S speed/accuracy
BATCH_SIZE = 32
VAL_SPLIT = 0.2
EPOCHS = 25

# 4) Build datasets (image_dataset_from_directory keeps labels and class_names consistent)
train_ds = tf.keras.utils.image_dataset_from_directory(
    IMAGE_DIR,
    labels="inferred",
    label_mode="categorical",
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    IMAGE_DIR,
    labels="inferred",
    label_mode="categorical",
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)
with open(os.path.join(MODEL_DIR, "class_names.txt"), "w") as f:
    for c in class_names:
        f.write(c + "\n")

# 5) Performance options
def configure(ds, training=False):
    if training:
        ds = ds.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
    return ds.prefetch(AUTOTUNE)

# 6) Augment and preprocess
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="augmentation")

preprocess = tf.keras.applications.efficientnet_v2.preprocess_input

def prep_batch(x, y, training=False):
    x = tf.cast(x, tf.float32)
    if training:
        x = augment(x, training=True)
    x = preprocess(x)
    return x, y

train_ds = train_ds.map(lambda x, y: prep_batch(x, y, training=True), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: prep_batch(x, y, training=False), num_parallel_calls=AUTOTUNE)
train_ds = configure(train_ds, training=True)
val_ds = configure(val_ds, training=False)

# 7) Class weights (from directory counts to avoid long dataset scans)
def count_images_per_class(root, class_names):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    counts = []
    for cname in class_names:
        cpath = os.path.join(root, cname)
        cnt = 0
        for fn in os.listdir(cpath):
            if fn.lower().endswith(exts):
                cnt += 1
        counts.append(cnt)
    return counts

counts = count_images_per_class(IMAGE_DIR, class_names)
indices = np.arange(num_classes)
all_labels = np.concatenate([np.repeat(i, counts[i]) for i in range(num_classes)])
cw = class_weight.compute_class_weight('balanced', classes=indices, y=all_labels)
class_weight_dict = {i: float(cw[i]) for i in indices}
print("Class weights:", class_weight_dict)

# 8) Build single SOTA model: EfficientNetV2-S
def build_model(img_size, num_classes):
    base = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        input_shape=(img_size, img_size, 3),
        weights="imagenet"
    )
    # Fine-tune top layers only for stability and speed
    for layer in base.layers[:-80]:
        layer.trainable = False
    for layer in base.layers[-80:]:
        layer.trainable = True

    inputs = tf.keras.Input((img_size, img_size, 3))
    # IMPORTANT: do not do preprocessing here again; already done in pipeline
    x = base(inputs, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(384, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="EffNetV2S_bovine")
    return model

model = build_model(IMG_SIZE, num_classes)
model.summary()

# 9) Optimizer, loss, schedules, callbacks
steps_per_epoch = int(np.ceil(train_ds.cardinality().numpy()))
total_steps = steps_per_epoch * EPOCHS
cosine = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=total_steps, alpha=0.1
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=cosine),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

ckpt_path = os.path.join(MODEL_DIR, "effnetv2s_best.keras")
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1
    ),
]

# 10) Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# 11) Evaluate
best = tf.keras.models.load_model(ckpt_path)
val_loss, val_acc = best.evaluate(val_ds, verbose=0)
print(f"Final validation accuracy: {val_acc*100:.2f}%")

# 12) Save final
best.save(os.path.join(MODEL_DIR, "effnetv2s_final.keras"))
print("Saved:", os.path.join(MODEL_DIR, "effnetv2s_final.keras"))

# 13) Save history for external visualization
np.save(os.path.join(MODEL_DIR, 'history.npy'), history.history)
print("Saved training history to history.npy")
