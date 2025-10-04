# Final Submission Merge + Evaluate (robust)
import sys, os, traceback
import numpy as np

def merge_six_into_one(output_file="CP1_submission_SM_BA.npy"):
    problem_files = [f"problem{i}.npy" for i in range(1, 7)]
    final_payload = {}

    for i, fname in enumerate(problem_files, start=1):
        key = f"Problem {i}"
        try:
            data = np.load(fname, allow_pickle=True).item()
        except Exception as e:
            print(f"⚠️ Could not load {fname}: {e}", flush=True)
            final_payload[key] = []
            continue

        if key not in data:
            # If a full dict was saved but with different keys, show them for debugging
            print(f"⚠️ {fname} missing {key}. Keys present: {list(data.keys())}", flush=True)
            final_payload[key] = []
        else:
            final_payload[key] = data[key]
            print(f"✓ Added {key} from {fname} ({len(final_payload[key])} mechanisms)", flush=True)

    np.save(output_file, final_payload, allow_pickle=True)
    print(f"✅ Final submission saved to {output_file}", flush=True)
    return output_file

def evaluate_file(npy_path):
    print("🔍 Importing evaluator...", flush=True)
    try:
        from LINKS.CP import evaluate_submission
    except Exception as e:
        print("❌ Failed to import evaluate_submission from LINKS.CP", flush=True)
        traceback.print_exc()
        return None

    print("🔍 Loading submission dict...", flush=True)
    try:
        submission = np.load(npy_path, allow_pickle=True).item()
    except Exception as e:
        print(f"❌ Failed to load {npy_path}: {e}", flush=True)
        traceback.print_exc()
        return None

    # Basic sanity check on keys
    expected = [f"Problem {i}" for i in range(1, 7)]
    missing = [k for k in expected if k not in submission]
    if missing:
        print(f"⚠️ Submission missing keys: {missing}", flush=True)
        print(f"   Present keys: {list(submission.keys())}", flush=True)

    print("🧮 Evaluating...", flush=True)
    try:
        score = evaluate_submission(submission)
    except Exception as e:
        print("❌ evaluate_submission raised an exception:", flush=True)
        traceback.print_exc()
        return None

    print("\n=== CP1 Submission Score ===", flush=True)
    print(score, flush=True)           # should be a dict-like breakdown
    # If you want totals:
    try:
        # Sometimes the evaluator returns a dict with 'Total Score' / 'Score Breakdown'
        total = score.get("Total Score", None)
        if total is not None:
            print(f"\nTotal Score: {total}", flush=True)
    except Exception:
        pass
    return score

if __name__ == "__main__":
    out = merge_six_into_one("CP1_submission_SM_BA.npy")
    evaluate_file(out)
