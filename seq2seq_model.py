import tensorflow as tf
from tensorflow.python.layers.core import Dense

assert tf.__version__ == '1.6.0'


def grap_inputs() :
    
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name = 'targets')
    keep_probs = tf.placeholder(tf.float32, name='dropout_rate')
    
    encoder_seq_len = tf.placeholder(tf.int32, (None, ), name='encoder_seq_len')
    decoder_seq_len = tf.placeholder(tf.int32, (None, ), name='decoder_seq_len')
    # tf.reduce_max() decoder_seq_len の最大の要素を返す
    max_seq_len = tf.reduce_max(decoder_seq_len, name='max_seq_len')
    max_seq_len = tf.reduce_max(decoder_seq_len, name='max_seq_len')

    
    return inputs, targets, keep_probs, encoder_seq_len, decoder_seq_len, max_seq_len

# encoder

def encoder(inputs, rnn_size, number_of_layers, encoder_seq_len, keep_probs, encoder_embed_size, encoder_vocab_size) :
    
    def cell(units, probs) :
        layer = tf.contrib.rnn.BasicLSTMCell(units)
        return tf.contrib.rnn.DropoutWrapper(layer, probs)
    
    encoder_cell = tf.contrib.rnn.MultiRNNCell([cell(rnn_size, keep_probs) for _ in range(number_of_layers)])
    
    # inputs の 単語を全て　embedding している　
    encoder_embeddings = tf.contrib.layers.embed_sequence(inputs, encoder_vocab_size,  encoder_embed_size)
    
    # encoder_cell で指定されたリカレントニューラルネットワークを作成する   この部分を bidirectional_dynamic_rnn() にしたい
    encoder_outputs, encoder_states = tf.nn.dynamic_rnn(encoder_cell,
                                                                                                    encoder_embeddings,
                                                                                                    encoder_seq_len,
                                                                                                    dtype=tf.float32)
    
    return encoder_outputs, encoder_states


# attention_mechanism 

def attention_mech(rnn_size, keep_probs, encoder_outputs, encoder_states, encoder_seq_len, batch_size) :
    
    def cell(units, probs) :
        layer = tf.contrib.rnn.BasicLSTMCell(units)
        return tf.contrib.rnn.DropoutWrapper(layer, probs)
    
    decoder_cell = cell(rnn_size, keep_probs)
    
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                                                                  memory=encoder_outputs,
                                                                                                  memory_sequence_length=encoder_seq_len,
                                                                                                  name='BahdanauAttention')
    
    # tf.contrib.seq2seq.AttentionWrapper 整数形状のタプル　
    dec_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                                                          attention_mechanism,
                                                                                          rnn_size /2)
    
    # ここではLSTM（この場合は）のデコーダセルのzero_stateを使用しており、最後のエンコーダの状態の値をそれに送ります
    attention_zero = dec_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    enc_state_new = attention_zero.clone(cell_state=encoder_states[-1])
    
    return dec_cell, enc_state_new


# decoder_inputs_data の 前処理

"""
inputs

targets = encoded_answers
word_to_id =  word to id 辞書
batch_size = 64

outputs

preprocessed version of decoder  inputs

"""

def decoder_inputs_preprocessing(targets, word_to_id, batch_size) :
    
    # endings = tf.strided_slice()API で encoded_answers　の 最後尾の要素を削除している
    endings = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    
    # endings の先頭に　付けている。 <GO> を付けている
    return tf.concat([tf.fill([batch_size, 1], word_to_id['<GO>']), endings], 1)


# decoder 定義

"""
decoder_inputs = decoder_inputs_preprocessing(targets, word_to_id, batch_size)
vocab_size = len(vocab) として　class Chatbot で定義している
embedding = tf.nn.embedding_lookup で embed_layer　　decoder_inputs　の値を抜き出す。

"""

def decoder(decoder_inputs, enc_states, dec_cell, decoder_embed_size, vocab_size, 
                       dec_seq_len, max_seq_len, word_to_id, batch_size) :
    
    
    # embed_layer = 全 word分の 分散表現cell　を作る　そこにtf.random_uniform 変数で 値を入れる。
    # tf.random_uniform() 均一な分布から　ランダムな値を生成
    embed_layer = tf.Variable(tf.random_uniform([vocab_size, decoder_embed_size]))
    embedding = tf.nn.embedding_lookup(embed_layer, decoder_inputs)
    
    # Args: kernel_initializer = 重み行列の初期化　
    # tf.truncated_normal_initializer() 指定した範囲の正規分布からランダムな値を生成
    output_layer = Dense(vocab_size, kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1))
    
    # training_decoder
    with tf.variable_scope('decoder') :
        train_helper = tf.contrib.seq2seq.TrainingHelper(embedding, dec_seq_len)
        
        train_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, train_helper, enc_states, output_layer)
        
        
        
        # 作成したtrain_decoder実行部分
        train_dec_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder, impute_finished=True, 
                                                                                                                       maximum_iterations=max_seq_len)
    # inference_decoder    
    with tf.variable_scope('decoder', reuse=True) :
        
        # starting_id_ve を batch_size 分作ります。
        starting_id_vec = tf.tile(tf.constant([word_to_id['<GO>']], dtype=tf.int32), [batch_size], name='starting_id_vec')
        
        # 恐らく　embed_layer から　全ての　vocab の 分散表現データを受け取り、引数として受け取った　start_token と end_token 情報を保持している
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embed_layer, starting_id_vec, word_to_id['<EOS>'])
        
        # inference時に使用するdecoder
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, inference_helper, enc_states, output_layer)
        
        #  作成したinference_decoder実行部分
        inference_dec_output, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder, impute_finished=True, 
                                                                                                                              maximum_iterations=max_seq_len)
        
    return train_dec_outputs, inference_dec_output



# 損失 最適化　定義

"""
outputs

loss

training_opt  = optimizer with clipped gradients
"""

def opt_loss(outputs, targets, dec_seq_len, max_seq_len, learning_rate, clip_rate) :
    
    logits = tf.identity(outputs.rnn_output)
    
    mask_weights = tf.sequence_mask(dec_seq_len, max_seq_len, dtype=tf.float32)
    
    with tf.variable_scope('opt_loss') :
        #using sequence_loss to optimize the seq2seq model
        loss = tf.contrib.seq2seq.sequence_loss(logits,
                                                                                targets,
                                                                                   mask_weights)
        
        # Define_optimizer
        opt = tf.train.AdamOptimizer(learning_rate)
        
        # 下記の３行は勾配爆発を防ぐため clip_gradient　を定義するために書く
        gradients = tf.gradients(loss, tf.trainable_variables())

        clipped_grads, _ = tf.clip_by_global_norm(gradients, clip_rate)
        trained_opt = opt.apply_gradients(zip(clipped_grads, tf.trainable_variables()))

        
    return loss, trained_opt






class Chatbot(object):
    
    def __init__(self, learning_rate, batch_size, enc_embed_size, dec_embed_size, rnn_size, 
                 number_of_layers, vocab_size, word_to_id, clip_rate):
        
        tf.reset_default_graph()
        
        self.inputs, self.targets, self.keep_probs, self.encoder_seq_len, self.decoder_seq_len, max_seq_len = grap_inputs()
        
        
        enc_outputs, enc_states = encoder(self.inputs, 
                                          rnn_size,
                                          number_of_layers, 
                                          self.encoder_seq_len, 
                                          self.keep_probs, 
                                          enc_embed_size, 
                                          vocab_size)
        
        dec_inputs = decoder_inputs_preprocessing(self.targets, 
                                                  word_to_id, 
                                                  batch_size)
        
        
        decoder_cell, encoder_states_new = attention_mech(rnn_size, 
                                                          self.keep_probs, 
                                                          enc_outputs, 
                                                          enc_states, 
                                                          self.encoder_seq_len, 
                                                          batch_size)
        
        train_outputs, inference_output = decoder(dec_inputs, 
                                                  encoder_states_new, 
                                                  decoder_cell,
                                                  dec_embed_size, 
                                                  vocab_size, 
                                                  self.decoder_seq_len, 
                                                  max_seq_len, 
                                                  word_to_id, 
                                                  batch_size)
        
        self.predictions  = tf.identity(inference_output.sample_id, name='preds')
        
        self.loss, self.opt = opt_loss(train_outputs, 
                                       self.targets, 
                                       self.decoder_seq_len, 
                                       max_seq_len, 
                                       learning_rate, 
                                       clip_rate)