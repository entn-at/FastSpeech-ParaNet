import torch
import torch.nn as nn

# from transformer.Models import Encoder, Decoder
from transformer.Models import Encoder, newDecoder
from transformer.Layers import Linear, PostNet
# from transformer.Models import get_sinusoid_encoding_table
from Networks import LengthRegulator, DecoderPreNet, get_position
import hparams as hp


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder()
        self.decoder_prenet = DecoderPreNet()
        # self.length_regulator = LengthRegulator()
        self.decoder = newDecoder()

        self.mel_linear = Linear(hp.decoder_output_size, hp.num_mels)
        self.postnet = PostNet()

        # n_position = hp.max_sep_len + 1
        # self.position_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(
        #     n_position, hp.encoder_output_size, padding_idx=0), freeze=True)

    def forward(self, src_seq, src_pos, dec_pos=None):
        encoder_output, _ = self.encoder(src_seq, src_pos)

        if self.training:
            # dec_pos = get_position(src_pos)
            decoder_input = self.decoder_prenet(encoder_output, dec_pos)

            # print(dec_pos.size())
            # print(decoder_input.size())

            # decoder_output = self.decoder(decoder_input, encoder_output, dec_pos)
            decoder_output = self.decoder(
                decoder_input, encoder_output, src_pos, dec_pos)

            mel_output = self.mel_linear(decoder_output)
            mel_output_postnet = self.postnet(mel_output) + mel_output
        else:
            dec_pos = get_position(src_pos)
            decoder_input = self.decoder_prenet(encoder_output, dec_pos)

            decoder_output = self.decoder(
                decoder_input, encoder_output, src_pos, dec_pos)

            mel_output = self.mel_linear(decoder_output)
            mel_output_postnet = self.postnet(mel_output) + mel_output

        # if self.training:
        #     length_regulator_output, decoder_pos, duration_predictor_output = self.length_regulator(
        #         encoder_output,
        #         encoder_mask,
        #         length_target,
        #         alpha,
        #         mel_max_length)
        #     decoder_output = self.decoder(length_regulator_output, decoder_pos)

        #     mel_output = self.mel_linear(decoder_output)
        #     mel_output_postnet = self.postnet(mel_output) + mel_output

        #     return mel_output, mel_output_postnet, duration_predictor_output
        # else:
        #     length_regulator_output, decoder_pos = self.length_regulator(
        #         encoder_output, encoder_mask, alpha=alpha)

        #     decoder_output = self.decoder(length_regulator_output, decoder_pos)

        #     mel_output = self.mel_linear(decoder_output)
        #     mel_output_postnet = self.postnet(mel_output) + mel_output

        #     return mel_output, mel_output_postnet

        return mel_output, mel_output_postnet
