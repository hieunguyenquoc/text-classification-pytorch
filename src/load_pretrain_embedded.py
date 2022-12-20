import numpy as np

def load_pretrain_embedded():
    vocab = []
    embedding = []
    with open('glove.6B.50d.txt','rt', encoding="utf-8") as f:
        full_content = f.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word) 
        embedding.append(i_embeddings)
    
    vocab_np = np.array(vocab)
    embeds_np = np.array(embedding)

    #insert '<pad>' and '<unk>' tokens at start of vocab_npa.
    vocab_np = np.insert(vocab_np, 0, '<pad>')
    vocab_np = np.insert(vocab_np, 1, '<unk>')

    pad_emb_npa = np.zeros((1,embeds_np.shape[1]))   #embedding for '<pad>' token.
    unk_emb_npa = np.mean(embeds_np,axis=0,keepdims=True)    #embedding for '<unk>' token.

    #insert embeddings for pad and unk tokens at top of embs_npa.
    embeds_np = np.vstack((pad_emb_npa,unk_emb_npa,embeds_np))

    return embeds_np