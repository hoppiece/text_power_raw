from gensim.models import Word2Vec
from datasets import load_dataset, load_from_disk
import time

# WikiTextデータセットのロード
#dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
dataset = load_from_disk("../wikitext-alpha-1.5")

# テキストデータを文単位でリストに変換
sentences = [sentence.split() for sentence in dataset['text']]

# Word2Vecモデルの初期化と学習
start_time = time.time()

model = Word2Vec(sentences, vector_size=300, window=5, min_count=5, workers=48, sg=0)

# 学習時間の計測
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

# 学習済みモデルの保存
model.save("wikitext-alpha-1.5_word2vec.model")
