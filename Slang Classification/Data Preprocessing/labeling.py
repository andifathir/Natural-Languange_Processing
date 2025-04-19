import pandas as pd
import time
import os
from google import genai

# Set up the Gemini client
client = genai.Client(api_key="AIzaSyA7SBpB1oAJxs0tiSl6_wZeN7lUKv4MyuQ")

# File paths
input_file = "Slang Classification/Data Preprocessing/target_data.csv"
output_file = "chatgpt_prelabel.csv"
backup_file = "chatgpt_backup.csv"

# Load input data
df = pd.read_csv(input_file)

# Add 'label' column if not present
if 'label' not in df.columns:
    df['label'] = None

# Resume labeling if previously started
if os.path.exists(output_file):
    df = pd.read_csv(output_file)
    print("✅ Resuming from saved progress.")

def classify_batch(texts):
    prompt = """Tugas Anda adalah mengklasifikasikan setiap kalimat ke salah satu kategori berikut:
1. Casual Slang
2. Internet Slang
3. Abbreviation
4. Offensive Slang
5. No Slang

Berikan hasil dalam format:
1. <kategori>
2. <kategori>
...dst

Kalimat:\n"""

    for i, t in enumerate(texts, 1):
        prompt += f"{i}. \"{t}\"\n"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt.strip()
    )
    raw_output = response.text.strip()

    # Parse results line by line
    lines = raw_output.splitlines()
    labels = []
    for line in lines:
        if '.' in line:
            parts = line.split('.', 1)
            label = parts[1].strip()
            labels.append(label)
    return labels

# Batching logic
BATCH_SIZE = 5
RATE_LIMIT_DELAY = 4.1  # seconds

try:
    for start in range(0, len(df), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_df = df.iloc[start:end]

        # Skip if already labeled
        if batch_df['label'].notnull().all():
            continue

        texts = batch_df['Text'].tolist()

        try:
            labels = classify_batch(texts)
            for i, label in enumerate(labels):
                df.at[start + i, 'label'] = label
        except Exception as e:
            print(f"\n❌ Error classifying batch {start}-{end}: {e}")
            df.to_csv(output_file, index=False)
            break

        # Save backup
        if start % 20 == 0:
            df.to_csv(backup_file, index=False)
            print(f"💾 Saved backup at row {start}")

        time.sleep(RATE_LIMIT_DELAY)

    # Final save
    df.to_csv(output_file, index=False)
    print("\n🎉 Done! Final labeled file saved.")

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user. Saving progress...")
    df.to_csv(output_file, index=False)
