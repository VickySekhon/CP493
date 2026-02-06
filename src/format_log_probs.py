read_path = "data/log-probs/1.txt"
write_path = "data/log-probs/1_fixed.txt"

text = []
with open(read_path, "r", encoding="utf-8") as f:
     line = f.readline()
     splits = line.split("ChatCompletionTokenLogprob")
     splits = [l + "\n" for l in splits]
with open(write_path, "w", encoding="utf-8") as f2:
     f2.writelines(splits)