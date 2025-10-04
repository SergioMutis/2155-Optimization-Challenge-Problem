import numpy as np

# Names of your individual files
problem_files = [
    "problem1.npy",
    "problem2.npy",
    "problem3.npy",
    "problem4.npy",
    "problem5.npy",
    "problem6.npy",
]

# Load each and insert into final dict
final_payload = {}
for i, fname in enumerate(problem_files, start=1):
    key = f"Problem {i}"
    try:
        data = np.load(fname, allow_pickle=True)
        # Some people save a dict, others save a list — handle both
        if isinstance(data.item(), dict):
            # if saved as dict (e.g. {"Problem 1": [...]})
            inner = data.item()
            final_payload[key] = inner.get(key, inner)
        else:
            final_payload[key] = data.tolist()
    except Exception as e:
        print(f"⚠️ Could not load {fname}: {e}")
        final_payload[key] = []

# Save the combined submission
output_file = "CP1_submission_SM_BA.npy"
np.save(output_file, final_payload, allow_pickle=True)
print(f"✅ Final submission saved to {output_file}")
