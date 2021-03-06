import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
encoder_inputs = keras.layers.Input(shape=(None),dtype=np.int32)
decoder_inputs = keras.layers.Input(shape=(None),dtype=np.int32)
sequence_lengths = keras.layers.Input(shape=[],dtype=np.int32)

embeddings = keras.layers.Embedding(vocab_size, embed_size)
encoder_embeddings= embeddings(encoder_inputs)
decoder_embeddings=embeddings(decoder_inputs)

encoder = keras.layers.LSTM(512, return_state=True)
#최종 은닉 상태를 디코더로 보내기 위해(최종h) 상태를 저장한다. 
#LSTM 특성상 은닉상태 두개를 반환(단기와 장기) 
encoder_outputs, state_h, state_c = encoder(encoder_embeddings)
encoder_state = [state_h, state_c]


sampler = tfa.seq2seq.sampler.TrainingSampler()
#텐서플로 내장 샘플러중 하나. 각 스텝별로 디코더에게 이전 스텝의 출력이 무엇인지 알려줌.
#추론 시 실제로 출력되는 토큰의 임베딩.
#훈련시에는 이전 타깃 토큰의 임베딩. 
#실전에서는 이전 타임 스텝의 타깃의 임베딩을 사용해 훈련을 시작해 
#이전 타임 스텝의 실제 출력된 토큰의 임베딩으로 바꾸는 것이 좋다. 
decoder_cell = keras.layers.LSTMCell(512)
output_layer = keras.layers.Dense(vocab_size)
decoder = tfa.seq2seq.basic_decoder.BasicDecoder(decoder_cell, sampler, output_layer=output_layer)


final_outputs, final_state, final_sequence_lengths = decoder(
  decoder_embeddings, initial_state=encoder_state,
  sequench_length=sequence_lengths
)
Y_proba = tf.nn.softmax(final_outputs.rnn_output)
model = keras.Model(inputs=[encoder_inputs, decoder_inputs, sequence_lengths],
outputs=[Y_proba])

#beam width에 해당하는 케이스를 만들어, 첫번째 디코더 스텝에서 모델이 가능한 모든 언어에 대한 
#추정 확률을 출력함. 최상위에 돌라온 단어를 beam width에 해당하는 모델로 복사하여 각 문장의 다음 단어를 찾음.
#vocab에 있는 단어의 추정 확률을 출력한다. 
beam_width = 10#빔 너비 10개...! 
decoder = tfa.seq2seq.beam_search_decoder.BeamSearchDecoder(
  cell=decoder_cell, beam_width=beam_width, output_layer=output_layer) #10짜리 Beamsearch 디코더 선언
  decoder_initial_state = tfa.seq2seq.beam_search_decoder.tile_batch(#각 디코더를 위해 각 인코더의 마지막 상태를 복사
    encoder_state, multiplier=beam_width)
  outputs, _, _ = decoder(
    embedding_decoder, start_tokens=start_tokens, end_token=end_token,
    initial_state=decoder_initial_state)#시작토큰, 종료토큰, 그리고 이 인코더의 마지막 상태를 디코더에 전달한다.