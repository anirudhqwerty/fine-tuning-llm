import json

input_file = "dataset/mo_customer_support.json"
output_file = "dataset/final_dataset.jsonl"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_file, "w", encoding="utf-8") as f_out:
    for i, item in enumerate(data):
        prompt = f"Customer: {item['input'].strip()}\nAgent:"
        response = item["output"].strip()
        json.dump({"prompt": prompt, "response": response}, f_out, ensure_ascii=False)
        f_out.write("\n")

print("Conversion complete! Saved to", output_file)
