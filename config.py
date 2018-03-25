VOCAB_THRESHOLD = 10

BUCKETS = [(10, 15), (15, 25), (25, 45), (45, 60), (60, 100)] #First try buckets you can tweak these

EPOCHS = 2

BATCH_SIZE = 64

RNN_SIZE = 512   #  RNN cell の数 embedding（分散表現した単語）　を 入れるためのcell である。

NUM_LAYERS = 3   # 積み重ねる層の数

# 　encoderにおいて１単語を embedding(分散表現)　で分割する　dimension(次元数)
ENCODING_EMBED_SIZE = 512 

# 　decoderにおいて１単語を embedding(分散表現)　で分割する　dimension(次元数)
DECODING_EMBED_SIZE = 512    

LEANING_RATE = 0.0001  # 学習係数　　学習を一度にどこまで行うかを決める。

# drop_out率 を決める。 学習時にある程度学習したらあまり関係のなさそうなノードを消す割合の指定。大体　0.5 半分を消す。
# drop_out を行うのは学習時のみで　テスト時には　drop_out は行わないので　1.0 に設定する
KEEP_PROBS = 0.5

CLIP_RATE = 4  #  勾配爆発を防ぐために勾配の範囲をclip　する。