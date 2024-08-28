import gensim
import gensim.downloader as api
import numpy as np
from datasets import load_dataset, load_from_disk

model_path = "GoogleNews-vectors-negative300.bin.gz"
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

# WikiTextデータセットのロード
#dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
dataset = load_from_disk("../wikitext-alpha-2.0")

def embed_sentence(sentence):
    # 文の各単語をベクトルに変換し、平均を取る
    words = sentence.split()
    word_vectors = []
    for word in words:
        if word in word2vec_model:
            word_vectors.append(word2vec_model[word])
    
    if word_vectors:
        sentence_embedding = np.mean(word_vectors, axis=0)
    else:
        # もし文のすべての単語がモデルに存在しない場合、ゼロベクトルを使用
        sentence_embedding = np.zeros(300)
    
    return sentence_embedding

# 文埋め込みを計算
embeddings = np.array([embed_sentence(sentence) for sentence in dataset['text']])

# 埋め込みを保存する (例: numpyで保存)
#np.save("wikitext_word2vec_embeddings.npy", embeddings)
np.save("wikitext_word2vec-alpha-2.0_embeddings.npy", embeddings)

print(f"Total embeddings: {embeddings.shape[0]}, Embedding dimension: {embeddings.shape[1]}")
